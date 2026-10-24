"""Package projected factors in the format stage5_s170_readout.py reads (model "leafletfa").

    python export_for_s170.py --run <run dir> --obs <DS>_TEST_obs.npz --out <dir>/<DS>_leafletfa.npz

Keys: cells, label, mouse, planted (copied from the obs dump, same row order) and
splicing__<input> = centred log phi (n_cells x K) for every factors_<input>.npz in the run dir,
rows reordered to the obs dump's cell order. Refuses to overwrite.
"""
import argparse, glob, os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; not overwritten")
    o = np.load(a.obs, allow_pickle=True)
    cells = o["cells"].astype(str)
    out = {k: o[k] for k in ("cells", "label", "mouse", "planted")}
    pos = {c: i for i, c in enumerate(cells)}
    for f in sorted(glob.glob(os.path.join(a.run, "factors_*.npz"))):
        name = os.path.basename(f)[len("factors_"):-4]
        z = np.load(f, allow_pickle=True)
        cid = z["cell_id"].astype(str)
        if len(cid) != len(cells) or set(cid) != set(cells):
            raise SystemExit(f"{name}: cells differ from the obs dump ({len(cid)} vs {len(cells)})")
        phi = z["phi"][np.argsort([pos[c] for c in cid])]
        cl = np.log(np.clip(phi, 1e-6, None))
        out[f"splicing__{name}"] = (cl - cl.mean(1, keepdims=True)).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(f"wrote {a.out}: {sorted(k for k in out if '__' in k)}")


if __name__ == "__main__":
    main()
