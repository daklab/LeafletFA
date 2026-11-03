"""Check 10.3 diagnostics (read-only, Karin 2026-11-02; check 10 stands as reported): interval widths vs errors, examples,
and coverage three ways. No new fits; reads the saved seed-0 check 10 outputs (primary intervals: q(phi), q(psi) + mouse
bootstrap) and the TEST counts.

    python c10_diag_intervals.py      # -> c10_diag/{c10_diag_units.tsv.gz, c10_diag_coverage.tsv, *_review.png, *.pdf}

Units as in 10.3: (type, junction) with TEST reads at the ATSE in 100-499 or >= 500 and >= 10 covering TEST cells.
Coverage:
  (a) TEST pseudobulk psi inside the LeafletFA 95 % interval (as in 10.3);
  (b) EXPLORATORY: the LeafletFA interval overlaps the TEST pseudobulk's Jeffreys 95 % interval, Beta(y + 1/2, n - y + 1/2);
  (c) EXPLORATORY, predictive: the LeafletFA interval widened by the pseudobulk's binomial sampling variance at its read
      count: midpoint +- 1.96 sqrt((half-width / 1.96)^2 + psi_hat (1 - psi_hat) / n), clipped to [0, 1].
"""
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import beta
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis-bioinformatics/Bioinformatics2026/paper/figures")
import vlib  # noqa: E402
from c10_catalog import labels, fit_dir  # noqa: E402
import style  # noqa: E402

OUT = os.path.join(vlib.V, "c10_diag")
BINS = (("100-499", 100, 499), (">=500", 500, np.inf))


def units(ds):
    E = np.load(os.path.join(vlib.V, "c10", f"{ds}_s0_estimates.npz"), allow_pickle=True)
    z = np.load(os.path.join(fit_dir(ds, 0), "fit.npz"), allow_pickle=True)
    inv = np.asarray(z["inv_ev"], np.int64); ev = z["event_id"].astype(str); jid = z["junction_id"].astype(str)
    types = E["types"].astype(str)
    _, tt, _ = labels(ds, vlib.paths(ds)["test"])
    Y = vlib.read_counts(vlib.paths(ds)["test"]).tocsr()
    N = (Y @ sp.csr_matrix((np.ones(len(inv)), (np.arange(len(inv)), inv)), shape=(len(inv), len(ev)))).tocsr()
    rows = []
    for k, t in enumerate(types):
        it = np.flatnonzero(tt == t)
        if not len(it):
            continue
        yt = np.asarray(Y[it].sum(0)).ravel(); nt = np.asarray(N[it].sum(0)).ravel(); cv = np.asarray((N[it] > 0).sum(0)).ravel()
        for lab, lo_b, hi_b in BINS:
            j = np.flatnonzero((nt[inv] >= lo_b) & (nt[inv] <= hi_b) & (cv[inv] >= 10))
            if len(j):
                rows.append(pd.DataFrame(dict(dataset=ds, cell_type=t, event_id=ev[inv[j]], junction_id=jid[j], read_bin=lab,
                                              test_reads=nt[inv[j]].astype(int), test_junction_reads=yt[j].astype(int),
                                              test_cells_covering=cv[inv[j]].astype(int), psi=E["point"][k, j],
                                              lo=E["lo"][k, j], hi=E["hi"][k, j])))
    U = pd.concat(rows, ignore_index=True)
    y, n = U.test_junction_reads.values, U.test_reads.values
    U["psi_test"] = y / n
    U["jeff_lo"] = np.where(y > 0, beta.ppf(0.025, y + 0.5, n - y + 0.5), 0.0)
    U["jeff_hi"] = np.where(y < n, beta.ppf(0.975, y + 0.5, n - y + 0.5), 1.0)
    U["width"] = U.hi - U.lo; U["abs_err"] = (U.psi - U.psi_test).abs()
    mid, h = (U.lo + U.hi) / 2, (U.hi - U.lo) / 2
    hp = 1.96 * np.sqrt((h / 1.96) ** 2 + U.psi * (1 - U.psi) / n)
    U["pred_lo"], U["pred_hi"] = np.clip(mid - hp, 0, 1), np.clip(mid + hp, 0, 1)
    U["cov_a"] = (U.psi_test >= U.lo) & (U.psi_test <= U.hi)
    U["cov_b_overlap"] = (U.lo <= U.jeff_hi) & (U.jeff_lo <= U.hi)
    U["cov_c_predictive"] = (U.psi_test >= U.pred_lo) & (U.psi_test <= U.pred_hi)
    return U


def main():
    cfg = style.load_config(); style.apply_style(cfg); plt.rcParams["axes.unicode_minus"] = False
    c_l, c_t = style.color(cfg, "leafletfa"), style.color(cfg, "raw_reads")
    U = pd.concat([units(ds) for ds in ("TMS", "Allen")], ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    U.to_csv(os.path.join(OUT, "c10_diag_units.tsv.gz"), sep="\t", index=False, float_format="%.5g")
    S = []
    for (ds, b), X in U.groupby(["dataset", "read_bin"], sort=False):
        q = lambda v: np.quantile(v, [0.25, 0.5, 0.75])
        w, e = q(X.width), q(X.abs_err)
        S.append(dict(dataset=ds, read_bin=b, junctions=len(X), types=X.cell_type.nunique(),
                      width_q25=w[0], width_median=w[1], width_q75=w[2], abs_err_q25=e[0], abs_err_median=e[1], abs_err_q75=e[2],
                      coverage_a_as_10_3=X.cov_a.mean(), coverage_b_overlap_EXPLORATORY=X.cov_b_overlap.mean(),
                      coverage_c_predictive_EXPLORATORY=X.cov_c_predictive.mean()))
    S = pd.DataFrame(S); S.to_csv(os.path.join(OUT, "c10_diag_coverage.tsv"), sep="\t", index=False, float_format="%.4g")
    print(S.to_string(index=False))

    # 1 + 2: widths and |error| on one log axis, per dataset x read bin
    fig, axs = plt.subplots(2, 2, figsize=(183 * style.MM, 110 * style.MM), sharex=True)
    edges = np.logspace(-5, 0, 61)
    for r, ds in enumerate(("TMS", "Allen")):
        for c, (b, _, _) in enumerate(BINS):
            ax = axs[r, c]; X = U[(U.dataset == ds) & (U.read_bin == b)]
            for v, col, lab in ((X.width, c_l, "LeafletFA 95 % interval width"), (X.abs_err, c_t, "|LeafletFA - TEST pseudobulk ψ|")):
                vv = np.clip(v.values, 1e-5, None); m, (q1, q3) = np.median(v), np.quantile(v, [0.25, 0.75])
                ax.hist(vv, bins=edges, histtype="step", color=col, lw=1.0, label=f"{lab}: median {m:.3g} (IQR {q1:.3g}–{q3:.3g})")
                ax.axvline(max(m, 1e-5), color=col, lw=0.6, ls=(0, (3, 2)))
            ax.set_xscale("log"); ax.set_ylabel("Junctions"); ax.set_ylim(0, ax.get_ylim()[1] * 1.3)    # headroom for the legend
            ax.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]); ax.set_xticklabels(["0.00001", "0.0001", "0.001", "0.01", "0.1", "1"])
            ax.minorticks_off()
            ax.set_title(f"{ds}, {b} TEST reads (n = {len(X):,})", fontsize=7)
            ax.legend(loc="upper left", fontsize=5.5)
            if r == 1:
                ax.set_xlabel("Width or absolute difference (ψ units; values < 0.00001 shown at 0.00001)")
    fig.tight_layout(); style.save(cfg, fig, os.path.join(OUT, "c10_diag_width_vs_error"), formats=("pdf", "png"))

    # 3: 30 random (type, event) pairs per read bin, one random junction each, sorted by estimate
    rng = np.random.default_rng(0)
    fig, axs = plt.subplots(2, 2, figsize=(183 * style.MM, 150 * style.MM))
    for r, ds in enumerate(("TMS", "Allen")):
        for c, (b, _, _) in enumerate(BINS):
            ax = axs[r, c]; X = U[(U.dataset == ds) & (U.read_bin == b)]
            pairs = X[["cell_type", "event_id"]].drop_duplicates()
            pick = pairs.iloc[rng.choice(len(pairs), min(30, len(pairs)), replace=False)]
            Z = X.merge(pick, on=["cell_type", "event_id"]).groupby(["cell_type", "event_id"], group_keys=False) \
                 .apply(lambda g: g.iloc[[rng.integers(len(g))]]).sort_values("psi").reset_index(drop=True)
            yy = np.arange(len(Z))
            # the point estimate (fit's phi, psi) can fall outside its draw interval, so draw segment + marker, not errorbar
            ax.hlines(yy + 0.15, Z.lo, Z.hi, color=c_l, lw=0.8)
            ax.plot(Z.psi, yy + 0.15, "^", color=c_l, ms=2.5, label="LeafletFA (95 % interval)")
            ax.errorbar(Z.psi_test, yy - 0.15, xerr=[Z.psi_test - Z.jeff_lo, Z.jeff_hi - Z.psi_test], fmt="o", color=c_t, ms=2.5,
                        elinewidth=0.8, capsize=0, mfc="white", label="TEST pseudobulk (Jeffreys 95 %)")
            ax.set_yticks(yy); ax.set_yticklabels([f"{t[:28]} · {e.split('_atse_')[-1]}" for t, e in zip(Z.cell_type, Z.event_id)],
                                                  fontsize=4.5)
            ax.set_xlim(-0.02, 1.02); ax.set_xlabel("Junction usage ψ")
            ax.set_title(f"{ds}, {b} TEST reads: 30 random (type, event) pairs", fontsize=7)
            if r == 0 and c == 0:
                ax.legend(loc="lower right", fontsize=5.5)
    fig.tight_layout(); style.save(cfg, fig, os.path.join(OUT, "c10_diag_examples"), formats=("pdf", "png"))

    # 4: coverage three ways
    fig, ax = plt.subplots(figsize=(120 * style.MM, 60 * style.MM))
    cols = [("coverage_a_as_10_3", "(a) as in 10.3", c_l), ("coverage_b_overlap_EXPLORATORY", "(b) interval overlap, exploratory", style.color(cfg, "null_celltype")),
            ("coverage_c_predictive_EXPLORATORY", "(c) predictive, exploratory", style.color(cfg, "consensus"))]
    x = np.arange(len(S))
    for i, (k, lab, col) in enumerate(cols):
        ax.bar(x + (i - 1) * 0.26, S[k], 0.26, color=col, label=lab)
    ax.axhline(0.95, color=style.color(cfg, "null_k1"), lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks(x); ax.set_xticklabels([f"{d}\n{b} reads\n(n = {n:,})" for d, b, n in zip(S.dataset, S.read_bin, S.junctions)])
    ax.set_ylim(0, 1); ax.set_ylabel("Fraction of junctions covered")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=5.5, ncol=3)
    fig.tight_layout(); style.save(cfg, fig, os.path.join(OUT, "c10_diag_coverage"), formats=("pdf", "png"))


if __name__ == "__main__":
    main()
