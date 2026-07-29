"""phi_init must actually reach the guide -- a silent no-op here would be invisible downstream.

phi cannot be initialised through `init_to_value` the way psi is: it lives under a subsampling plate,
so at prototype-trace time the site is batch-shaped (B, K) and a full (N, K) value raises a shape
error. It is applied instead by writing AutoNormal's location parameter directly. That parameter is
(N, K-1) -- Dirichlet's unconstrained space is stick-breaking -- and does not exist until the guide
has been traced once.

Because `_apply_phi_init` skips gracefully when the parameter is missing or mis-shaped, a wrong key
would produce no error and no effect. Any experiment comparing "psi pinned" against "psi and phi
pinned" would then show no difference and be read as "phi's initialisation does not matter", which is
the wrong conclusion drawn from a bug. These tests exist to make that impossible.
"""
import numpy as np
import pyro
import pytest
import torch
from torch.distributions import biject_to, constraints

from leafletfa.minibatch import LeafletFAMinibatch


def _toy(N=200, J=60, K=4, seed=0):
    rng = np.random.default_rng(seed)
    clus = rng.integers(1, 25, (N, J)).astype("float32")
    junc = (clus * rng.random((N, J))).astype("float32")
    return junc, clus, K


def _structured_phi(N, K, first=0, second=2):
    """A phi with obvious structure, so a failure to apply it is visible rather than subtle."""
    p = np.full((N, K), 0.01, dtype="float32")
    p[: N // 2, first] = 0.97
    p[N // 2 :, second] = 0.97
    return p / p.sum(1, keepdims=True)


def test_phi_init_round_trips_through_the_guide_parameter():
    junc, clus, K = _toy()
    phi0 = _structured_phi(junc.shape[0], K)
    pyro.clear_param_store()
    m = LeafletFAMinibatch(junc, clus, K=K, phi_init=phi0, device="cpu")
    m._B = 32
    g = m._make_guide(32)
    g()
    m._apply_phi_init()
    back = biject_to(constraints.simplex)(
        pyro.get_param_store()["AutoNormal.locs.phi"]
    ).detach().numpy()
    assert np.abs(back - phi0).max() < 1e-4


def test_untrained_model_returns_phi_init():
    junc, clus, K = _toy()
    phi0 = _structured_phi(junc.shape[0], K)
    pyro.clear_param_store()
    m = LeafletFAMinibatch(junc, clus, K=K, phi_init=phi0, device="cpu")
    m.fit(num_steps=0, subsample_size=32, verbose=False, early_stop=False)
    got = m.phi_mean().cpu().numpy()
    assert (got.argmax(1) == phi0.argmax(1)).mean() > 0.99


def test_a_different_phi_init_gives_a_different_start():
    """Guards the case where phi_mean happens to look right for reasons unrelated to the init."""
    junc, clus, K = _toy()
    N = junc.shape[0]
    outs = []
    for a, b in ((0, 2), (1, 1)):
        pyro.clear_param_store()
        m = LeafletFAMinibatch(junc, clus, K=K, phi_init=_structured_phi(N, K, a, b), device="cpu")
        m.fit(num_steps=0, subsample_size=32, verbose=False, early_stop=False)
        outs.append(m.phi_mean().cpu().numpy())
    assert not np.allclose(outs[0], outs[1])


def test_default_init_is_unrelated_to_phi_init():
    """Without this the three tests above could all pass on a model that ignores phi_init entirely."""
    junc, clus, K = _toy()
    phi0 = _structured_phi(junc.shape[0], K)
    pyro.clear_param_store()
    m = LeafletFAMinibatch(junc, clus, K=K, device="cpu")
    m.fit(num_steps=1, subsample_size=32, verbose=False, early_stop=False)
    got = m.phi_mean().cpu().numpy()
    assert abs(np.corrcoef(got.ravel(), phi0.ravel())[0, 1]) < 0.5


def test_phi_init_shape_is_validated():
    junc, clus, K = _toy()
    with pytest.raises(AssertionError):
        LeafletFAMinibatch(junc, clus, K=K, phi_init=np.ones((7, K), dtype="float32"), device="cpu")
