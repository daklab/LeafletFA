"""leafletfa.catalog.celltype_psi reproduces the saved check 10 point estimates and the browser's TRAIN tables.

    python catalog_port_check.py <TMS|Allen>     # CPU, sbatch; -> $V/c12/export/<ds>_port_check.json; exits non-zero on failure

Pass: max |psi - c10 point| < 1e-4 over all types x junctions; ATSE reads, cells with reads, junction reads and mice equal
the browser's train_tables.npz exactly; raw psi equals junction reads / ATSE reads.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import vlib  # noqa: E402
import c10_catalog as C  # noqa: E402
from leafletfa.catalog import load_fit, celltype_psi  # noqa: E402

ds = sys.argv[1]; V = vlib.V
E = np.load(os.path.join(V, "c10", f"{ds}_s0_estimates.npz"), allow_pickle=True)
Tt = np.load(os.path.join(V, "portal_browser", "precomputed", f"{ds}_train_tables.npz"))
fit = load_fit(C.fit_dir(ds, 0))
P = vlib.paths(ds)["train"]
cells, t, mouse = C.labels(ds, P)
assert np.array_equal(cells, fit["cell_id"])
Y = vlib.read_counts(P).tocsr()
types = list(E["types"].astype(str))
out = celltype_psi(Y, fit, t, types=types, mice=mouse)
d = float(np.abs(out["psi"] - E["point"]).max())
chk = dict(
    dataset=ds, types=len(types), junctions=int(out["psi"].shape[1]),
    max_abs_diff_point=d,
    atse_reads_equal=bool(np.array_equal(out["atse_reads"], Tt["ereads"].astype(np.float64))),
    cells_covering_equal=bool(np.array_equal(out["cells_covering"], Tt["cov"].astype(np.float64))),
    junction_reads_equal=bool(np.array_equal(out["junction_reads"], Tt["jreads"].astype(np.float64))),
    mice_equal=bool(np.array_equal(out["mice"], Tt["mice"].astype(np.int64))),
    cells_equal=bool(np.array_equal(out["cells"], E["cells"].astype(np.float64))),
)
tot = Tt["ereads"][:, fit["inv"]].astype(np.float64)
with np.errstate(invalid="ignore", divide="ignore"):
    raw = np.where(tot > 0, Tt["jreads"] / tot, np.nan)
chk["raw_psi_equal"] = bool(np.allclose(out["raw_psi"], raw, equal_nan=True, rtol=0, atol=1e-12))
chk["pass"] = bool(d < 1e-4 and all(v for k, v in chk.items() if k.endswith("_equal")))
os.makedirs(os.path.join(V, "c12", "export"), exist_ok=True)
json.dump(chk, open(os.path.join(V, "c12", "export", f"{ds}_port_check.json"), "w"), indent=1)
print(json.dumps(chk, indent=1), flush=True)
sys.exit(0 if chk["pass"] else 1)
