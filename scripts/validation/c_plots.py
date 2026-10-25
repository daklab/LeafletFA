"""Validation summary plots (CPU) from the check TSVs -> <V>/plots/validation_summary.{pdf,png}.

    python c_plots.py

a  check 1: held-out gain per read over each TRAIN null (95 % CI over mice), TMS and Allen
b  check 2: kNN (real, content-only) vs K, and the prior floored at 1
c  check 2: live programs and participation ratio vs K
d  check 3/2: programs replicating across seeds (centred psi, liveness) vs K, three grids
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402
sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/_common")
from leaflet_theme import apply_theme, PALETTE, figsize, panel_label, save  # noqa: E402


def main():
    import matplotlib.pyplot as plt
    apply_theme()
    T = os.path.join(vlib.V, "tsv")
    fig, ax = plt.subplots(1, 4, figsize=figsize("double", h=2.2))

    lab = {"null_i_population": "population ψ", "null_ii_celltype": "cell-type ψ", "null_iii_K1": "K = 1"}
    for x0, (ds, col) in enumerate((("TMS", PALETTE["teal"]), ("Allen", PALETTE["indigo"]))):
        g = pd.read_csv(os.path.join(T, f"c1_{ds}_K20_s0_gains.tsv"), sep="\t")
        g = g[g.comparison.str.replace("model_split - ", "").isin(lab)]
        for i, (_, r) in enumerate(g.iterrows()):
            y = i + 0.18 * (x0 * 2 - 1)
            ax[0].errorbar(r.gain_per_read * 1e3, y, xerr=[[(r.gain_per_read - r.ci_lo) * 1e3], [(r.ci_hi - r.gain_per_read) * 1e3]],
                           fmt="o", ms=3, color=col, elinewidth=0.8, capsize=0, label=ds if i == 0 else None)
        ax[0].set_yticks(range(len(lab))); ax[0].set_yticklabels([lab[c.replace("model_split - ", "")] for c in g.comparison])
    ax[0].axvline(0, color=PALETTE["grey"], lw=0.6, ls=(0, (3, 2)))
    ax[0].set_xlabel("Held-out gain (10$^{-3}$ nats per read)"); ax[0].legend(frameon=False, loc="lower right")

    k = pd.read_csv(os.path.join(T, "c2_K_prior.tsv"), sep="\t")
    kk = k[k.config.str.match(r"^K\d+$")].sort_values("K"); pf = k[k.config == "K20_priorfloor1"]
    ax[1].plot(kk.K, kk.knn_real, "-o", ms=3, color=PALETTE["teal"], lw=1.0, label="real")
    ax[1].plot(kk.K, kk.knn_contentonly, "-o", ms=3, color=PALETTE["rose"], lw=1.0, label="content-only")
    ax[1].plot(pf.K, pf.knn_real, "D", ms=3.5, color=PALETTE["teal"], mfc="white", label="prior ≥ 1, real")
    ax[1].plot(pf.K, pf.knn_contentonly, "D", ms=3.5, color=PALETTE["rose"], mfc="white", label="prior ≥ 1, content-only")
    ax[1].set_xlabel("K"); ax[1].set_ylabel("kNN accuracy (TMS test)"); ax[1].set_ylim(0, 0.6)
    ax[1].legend(frameon=False, fontsize=5, loc="lower right")

    ax[2].plot(kk.K, kk.live_programs, "-o", ms=3, color=PALETTE["teal"], lw=1.0, label="programs in use")
    ax[2].plot(kk.K, kk.participation_ratio, "-o", ms=3, color=PALETTE["ochre"], lw=1.0, label="participation ratio")
    ax[2].plot(kk.K, kk.K, color=PALETTE["grey"], lw=0.6, ls=(0, (3, 2)), label="K")
    ax[2].set_xlabel("K"); ax[2].set_ylabel("Dimensions"); ax[2].set_ylim(0, 31)
    ax[2].legend(frameon=False, fontsize=5, loc="upper left")

    for tag, col, name in (("TMS70", PALETTE["teal"], "TMS"), ("ABonly", PALETTE["indigo"], "mouse Allen"), ("atlas", PALETTE["olive"], "atlas")):
        s = pd.read_csv(os.path.join(T, f"c2_replication_{tag}_summary.tsv"), sep="\t")
        ax[3].plot(s.K, s.replicating, "-o", ms=3, color=col, lw=1.0, label=name)
    ax[3].set_xlabel("K"); ax[3].set_ylabel("Programs replicating\n(3 seeds, centred ψ ≥ 0.8)"); ax[3].set_ylim(0, 8)
    ax[3].legend(frameon=False, fontsize=5)
    for a, l in zip(ax, "abcd"):
        panel_label(a, l)
    fig.tight_layout()
    os.makedirs(os.path.join(vlib.V, "plots"), exist_ok=True)
    save(fig, os.path.join(vlib.V, "plots", "validation_summary"))
    print("saved", os.path.join(vlib.V, "plots", "validation_summary"))


if __name__ == "__main__":
    main()
