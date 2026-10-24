"""Sanity readout on projected LeafletFA factors: finite, and do cell types separate?

    python knn_check.py --run <run dir> --obs <DS_TEST_obs.npz> [--obs-h5mu <test h5mu>]

kNN (k 15) on centred log phi (the §170 embedding), standardised, mouse-grouped 4-fold, label =
tissue | broad cell type (`label` in the §170 obs dump). Reported per input with a label-permuted
baseline (same folds) and the majority-class rate. Writes knn_check.tsv in the run dir.
"""
import argparse, glob, os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--obs", required=True, help="npz with cells, label, mouse (the §170 obs dump)")
    ap.add_argument("--k", type=int, default=15)
    a = ap.parse_args()
    from sklearn.model_selection import GroupKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score

    o = np.load(a.obs, allow_pickle=True)
    obs = pd.DataFrame({"label": o["label"].astype(str), "mouse": o["mouse"].astype(str)},
                       index=o["cells"].astype(str))
    rows = []
    for f in sorted(glob.glob(os.path.join(a.run, "factors_*.npz"))):
        name = os.path.basename(f)[len("factors_"):-4]
        z = np.load(f, allow_pickle=True)
        phi, cells = z["phi"], z["cell_id"].astype(str)
        lab, mouse = obs.loc[cells, "label"].values, obs.loc[cells, "mouse"].values
        cl = np.log(np.clip(phi, 1e-6, None)); X = cl - cl.mean(1, keepdims=True)
        rng = np.random.default_rng(0)

        def score(y):
            pred = np.empty(len(y), dtype=object)
            for tr, te in GroupKFold(4).split(X, y, mouse):
                sc = StandardScaler().fit(X[tr])
                knn = KNeighborsClassifier(min(a.k, len(tr))).fit(sc.transform(X[tr]), y[tr])
                pred[te] = knn.predict(sc.transform(X[te]))
            return float((pred == y).mean()), float(f1_score(y, pred.astype(str), average="weighted"))
        acc, f1 = score(lab)
        acc0, f10 = score(rng.permutation(lab))
        rows.append(dict(input=name, n_cells=len(cells), n_labels=len(set(lab)),
                         finite=bool(np.isfinite(phi).all()),
                         rows_sum_to_1=bool(np.allclose(phi.sum(1), 1, atol=1e-4)),
                         live_programs=int(((np.bincount(phi.argmax(1), minlength=phi.shape[1]) / len(phi)) >= 0.01).sum()),
                         knn_acc=round(acc, 3), knn_f1w=round(f1, 3),
                         permuted_acc=round(acc0, 3), permuted_f1w=round(f10, 3),
                         majority=round(pd.Series(lab).value_counts(normalize=True).iloc[0], 3),
                         restarts=int(z["n_nan_restarts"])))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.run, "knn_check.tsv"), sep="\t", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
