## Check 19 — Allen pairs: descriptive all-pairs export and within-donor confirmation on pairs with ≥ 4 shared donors (amendment 15; `c19_allen_pairs.py`, `$V/c19/`)

**Jobs.**
- Export 23370837: 1.3 min, peak 1.1 GB.
- Smoke 23370838 (20 pairs): 33 min, peak 20.7 GB.
- Full 23371384: 54 min, peak 21.9 GB, 4 CPUs (about 3.6 CPU-h). 48 GB were requested; the combo count was unknown before the run.
- **Deviation:** the smoke log's tail, including two rows of its differences table and its verdict line, was seen
  while reading its status. The design was fixed before this and nothing was changed after it.

**Part 1, descriptive (no rule).**
- 2,278 pairs. Per pair, the top 200 ATSEs by |Δψ_lead|; median 10,588 eligible ATSEs per pair.
- Outputs: `allen_pairs_top200.tsv.gz` and `allen_pairs_index.tsv`.

| Donors with TRAIN cells of both groups | 0 (flagged "cell type and donor not separable; differences may reflect donor or batch") | 1 | 2–3 | ≥ 4 |
|---|---|---|---|---|
| Pairs | **1,094** | 216 | 326 | 642 |

**Part 2, check 19.**
- 642 pairs; 564 with evaluable units. Pair kinds: 405 same class, 237 different class.
- Shared donors per pair: 4–5: 202; 6–9: 150; 10–19: 121; ≥ 20: 169.
- 2,796 (group, donor) combinations.

| Bin (TRAIN reads, shared donors) | Pairs | LeafletFA excess (CI) | Raw excess (CI) | LeafletFA − raw (CI) |
|---|---|---|---|---|
| 1–19 | 525 | 0.084 (0.065, 0.103) | 0.054 (0.036, 0.072) | **+0.029 (+0.010, +0.049)** |
| 20–99 | 564 | 0.296 (0.275, 0.316) | 0.206 (0.185, 0.226) | **+0.091 (+0.070, +0.112)** |
| 100–499 | 564 | 0.452 (0.436, 0.469) | 0.392 (0.373, 0.413) | **+0.060 (+0.045, +0.076)** |
| ≥ 500 | 563 | 0.587 (0.569, 0.605) | 0.524 (0.505, 0.542) | **+0.064 (+0.053, +0.076)** |

**Pre-registered reading: BETTER.**
- All three compared lower bounds are > 0.
- Thin data (1–19 reads) supported: LeafletFA's excess lower bound is 0.065, and its lead over raw has a lower bound of
  +0.010.

**By pair kind (descriptive).** LeafletFA − raw (CI):

| Bin | Same class (≤ 395 pairs) | Different class (≤ 169 pairs) |
|---|---|---|
| 1–19 | +0.011 (−0.014, +0.034) | +0.073 (+0.040, +0.108) |
| 20–99 | +0.057 (+0.031, +0.082) | +0.170 (+0.135, +0.204) |
| 100–499 | +0.070 (+0.049, +0.091) | +0.037 (+0.020, +0.056) |
| ≥ 500 | +0.080 (+0.065, +0.095) | +0.025 (+0.014, +0.037) |

- Same class at 1–19 reads: not distinguishable from raw. This matches check 17's same-cell-type pairs.

**Rank bands (descriptive), LeafletFA excess by rank within pair.**

| Rank | 1–19 reads | 20–99 | 100–499 | ≥ 500 |
|---|---|---|---|---|
| 1–5 | 0.13 | 0.31 | 0.48 | 0.69 |
| 6–20 | 0.12 | 0.25 | 0.41 | 0.66 |
| 21–100 | 0.16 | 0.17 | 0.29 | 0.56 |

Full table: `rank_bands.tsv`. Uncapped, so not matched between methods.

**Limits stated in advance.**
- The seed-0 Allen fit saw all TRAIN cells. The TEST cells used for confirmation are held out per cell, but come from
  the same donors.
- Pairs share groups, so the pair-bootstrap intervals are approximate.
- The 1,636 pairs with fewer than 4 shared donors are not covered by check 19.
