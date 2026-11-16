"""Check 17, Allen part (amendment 13; report only): donors per pair of Allen cell groups contributing cells of both groups.
    python c17_allen_donors.py   -> $V/c17/allen_donors_per_pair.tsv, allen_donors_summary.tsv
Cell groups as check 10 (cluster label without its number, >= 10 TRAIN cells). Donor halves: the 10.2a split (seed 0 over TRAIN donors)."""
import itertools, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402
from c10_catalog import labels, MIN_TRAIN  # noqa: E402

OUT = os.path.join(vlib.V, "c17")


def main():
    os.makedirs(OUT, exist_ok=True); P = vlib.paths("Allen")
    _, t, m = labels("Allen", P["train"]); _, tt, tm = labels("Allen", P["test"])
    n = pd.Series(t).value_counts(); groups = sorted(n[n >= MIN_TRAIN].index); assert len(groups) == 68, len(groups)
    mice = np.unique(m); mA = set(np.random.default_rng(0).permutation(mice)[: len(mice) // 2])   # = mouse_halves(D)
    sets = {}
    for nm, (tl, ml, msk) in {"train": (t, m, None), "train_halfA": (t, m, "A"), "train_halfB": (t, m, "B"), "test": (tt, tm, None)}.items():
        keep = np.ones(len(tl), bool) if msk is None else (np.isin(ml, list(mA)) if msk == "A" else ~np.isin(ml, list(mA)))
        df = pd.DataFrame(dict(g=tl[keep], d=ml[keep])); sets[nm] = df.groupby("g").d.agg(set).to_dict()
    rows = [dict(A=a, B=b, **{f"donors_both_{k}": len(s.get(a, set()) & s.get(b, set())) for k, s in sets.items()})
            for a, b in itertools.combinations(groups, 2)]
    T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "allen_donors_per_pair.tsv"), sep="\t", index=False)
    bins = [-1, 0, 1, 3, np.inf]; labs = ["0", "1", "2-3", ">=4"]
    S = pd.DataFrame({k: pd.cut(T[f"donors_both_{k}"], bins, labels=labs).value_counts().reindex(labs) for k in sets}); S.index.name = "donors_with_both_groups"
    S.loc["median"] = [T[f"donors_both_{k}"].median() for k in sets]; S.loc["pairs"] = len(T)
    S.to_csv(os.path.join(OUT, "allen_donors_summary.tsv"), sep="\t"); print(S.to_string()); print(f"donors {len(mice)}, half A {len(mA)}")


if __name__ == "__main__":
    main()
