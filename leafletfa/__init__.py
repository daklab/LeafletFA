from leafletfa.model import LeafletFA
from leafletfa.minibatch import LeafletFAMinibatch

__version__ = "0.2.0"
__all__ = [
    "LeafletFA",
    "LeafletFAMinibatch",             # the shipped Dirichlet-multinomial model (use .dm(...))
    "estep",                          # infer phi for new cells against a frozen fit
    "dirichlet_multinomial",          # dm_log_prob, segment_softmax, denominator_consistency
    "waypoints",
    "differential_splicing",          # factor-level DS (program characterisation)
    "differential_splicing_groups",   # group-vs-group DS (the primary marker analysis)
    "estimate_bayesian_fdr",
    "utils",
]

_LAZY = (
    "estep",
    "dirichlet_multinomial",
    "waypoints",
    "differential_splicing",
    "differential_splicing_groups",
    "estimate_bayesian_fdr",
    "utils",
)


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return importlib.import_module(f"leafletfa.{name}")
    raise AttributeError(f"module 'leafletfa' has no attribute {name!r}")
