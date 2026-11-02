# Amendment 7 (DRAFT, 2026-11-02; not approved, not run) — predictive and recalibrated cell-type intervals

Status: draft for Karin's approval. Nothing below has been computed. Once approved, this text is appended unchanged to
`VALIDATION_PREREG.md` before any code runs. The check 10 results stand: 10.3 is reported as is, and no 10.1, 10.2 or 10.7
result or verdict changes.

**Question.** The check 10 intervals covered TEST pseudobulk ψ in 14–35 % of junctions against a nominal 95 %. Do they
reach nominal coverage when the interval also includes (a) the sampling noise of the pseudobulk being compared and (b) a
model-error term estimated on held-out mice?

**What the interval covers.** Both (a) and (a)+(b) are *predictive* intervals for the pseudobulk ψ of new cells of the type,
at a given read depth. They are not intervals for the true cell-type ψ. The paper text and the catalog columns will say so.

**Fixed inputs (unchanged from check 10).**
- Seed-0 fits, TRAIN estimator and point estimates as in check 10.
- The 200 combined draws, regenerated with the check 10 seeds: q(φ), q(ψ) plus the mouse bootstrap of note 3. As a
  variant, the parameter-only draws of note 2 are also used.
- Units: the 10.3 units, i.e. (type, junction) with ≥ 100 TEST reads and ≥ 10 covering TEST cells at the ATSE. TEST read
  bins are 100–499 and ≥ 500.
- Mouse halves A/B: the 10.2a split (seed 0; Allen donors likewise), applied to TEST cells by their `mouse.id`.
- In each half, a unit is used only if it meets the 10.3 read and cell rule within that half's TEST cells.

## (a) Pseudobulk sampling noise added to the parameter interval
- For each unit, draw s ∈ 1..200 and replicate r ∈ 1..5:
  - Take the draw's type ψ_t^(s) (the type mean of ρ̃ in that draw).
  - For every TEST cell c of the type in the comparison set, simulate its junction counts at that ATSE from
    **DM(N_ce, γ_e ψ_t^(s))**, using the cell's observed TEST read total N_ce and the fitted γ_e.
  - Sum over cells and form the simulated pseudobulk ψ.
- Interval (a): 2.5–97.5 % of the 1,000 simulated values.
- Sensitivity variant, reported beside it: a **binomial** at the pooled read count, which ignores between-cell
  overdispersion.
- No parameters are estimated from TEST, so (a) is evaluated on all TEST cells and also within each mouse half.

## (b) Model-error term, estimated on one mouse half and evaluated on the other
- For each unit in the calibration half, take the mean m and variance v of its (a) predictive distribution.
- Model-error variance: τ²_{d,b}, one per dataset d and TEST read bin b. It is the smallest τ² on the grid
  {0, 1e-5 · 2^i, i = 0..20} for which m ± 1.96 √(v + τ²) covers the calibration half's TEST pseudobulk ψ in ≥ 95 % of
  that bin's junctions.
- Interval (a)+(b): m ± 1.96 √(v + τ²_{d,b}), clipped to [0, 1], applied to the **other** half's units.
- Both directions are run: calibrate on A and evaluate on B, then calibrate on B and evaluate on A.

## Reported
- Coverage by dataset × TEST read bin for:
  - the check 10 interval (as in 10.3)
  - (a) DM
  - (a) binomial
  - (a)+(b), in each direction
- The same table for the parameter-only draws.
- Median interval width for each method, and τ²_{d,b} with √τ².
- Coverage of (a)+(b) by number of mice of the type (1–2, 3–5, ≥ 6) and by TRAIN reads (< 1,000, ≥ 1,000), descriptive.

## Pass (written before any run)
- **Threshold:** coverage within 0.05 of nominal, i.e. in [0.90, 1.00], at ≥ 100 TEST reads.
- **(a)+(b) passes** if this holds in **both** read bins (100–499, ≥ 500), **both** datasets and **both** directions,
  using the primary (combined) draws and the DM version of (a).
- (a) alone is checked against the same threshold and reported as pass or fail. It is not a gate.

## Readings (stated now)
- **(a) alone passes:** the under-coverage is explained by comparing a parameter interval to a noisy pseudobulk. The
  catalog ships the (a) interval, labeled as predictive at the stated read depth.
- **(a)+(b) passes but (a) fails:** the catalog ships the (a)+(b) interval. The text reports τ² as the model-error term
  and says it was estimated on held-out mice.
- **Both fail:** the catalog ships the check 10 interval, labeled "model-based, not calibrated", with the measured
  coverage. The paper makes no interval-coverage claim.
- In every case the paper does not claim calibrated intervals for the true cell-type ψ (Karin, 2026-11-02).

## Limits stated in advance
- TEST and TRAIN share mice (per-cell split). The mouse halves separate calibration from evaluation for (b) only.
  Both halves' estimates still use all TRAIN cells.
- Junctions of one ATSE and units of one type are not independent, so coverage is reported without a CI.

**Cost and code.**
- CPU only, from the fit checkpoints and TEST counts; no refit. About 1 h per dataset (same draw cost as check 10,
  without the null permutations).
- Code: `scripts/validation/c12_intervals.py` (to be written after approval). Results: `tsv/c12_*` and section
  "Check 12" in VALIDATION_REPORT.md.
