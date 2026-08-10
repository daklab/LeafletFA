# Next session — 2026-08-11

## First thing: the configuration decision

Both arms were training overnight. Score them and apply the rule in
`Leaflet-analysis/GenomeBiology_Submission2026/GAMEPLAN_final_models_and_figures.md`.

    ATLAS_DM/runs_final/            delta = K  +  m/(m-1) weight   (7/9 done at handoff)
    ATLAS_DM/runs_final_deltaonly/  delta = K  alone               (0/9 done, ~2-3.5h in)

**The rule: ship delta-only unless it CLEARLY fails.** Not "unless the weighted arm wins by a
margin". Two reasons:

1. The posterior IS the product. The model-based DS test is a posterior test, so a tempered
   likelihood forces permutation calibration forever.
2. The weight points the wrong way. It restores the BetaBinomial's over-count (m junctions treated
   as m independent observations when only m-1 are free) while the measured effective sample size is
   LOWER than the read count, not higher. `conc` already carries that correction.

Score with:

    A54_score_both_properties.py --fits 'label=<fit.npz>' ...

Weighted arm at handoff: 5 of 6 scored fits recover the Fbln2 direction (fibroblast < 0.1 against
LeafCutter's 0.020), versus 1 of 9 for the previous production run. One seed still collapses.

## The result that changed the model's justification

`conc` is an EFFECTIVE SAMPLE SIZE, and we measured it. Reads are not independent draws: on ATSEs
whose pooled composition within microglia is genuinely mixed, 23.8% of cells put ALL reads on one
junction at 50-198 reads, where independence predicts ~never. Solving P(all on one) = p^M + (1-p)^M
gives 1.66-3.07 effective molecules as reads go 6 -> 124. conc_min = 2.0 sits inside that range; the
free fit's 0.52 is fewer than one molecule. Written up in `METHODS_dirichlet_multinomial.md`.

## What phi can be claimed to be (A58/A59/A60)

- **8 of 20 programs replicate across three seeds at r 0.978-0.998.** Six sit at 0.7-0.9, six are
  degenerate (program 15 at -0.059). The distribution is bimodal, so an average over programs is
  meaningless -- report per-program replication and interpret only the replicating ones.
- **Averaging cells does NOT help.** A60 found across-seed agreement flat from n=1 to n=1000. The
  disagreement is systematic, not noise. The "noisy per cell, reliable in aggregate" framing is
  WRONG and should not be repeated.
- phi is dominated by cell type (92.8% of variance) not donor (7.2%), so programs are not secretly
  encoding mouse identity.

**Describe the filter accurately.** It is a POST HOC selection: fit K=20, then keep the programs that
survive across seeds. Say that, rather than "the model finds 8 programs". What makes it defensible
rather than cherry-picking is (a) the threshold is on REPLICATION, not on the result -- programs are
selected by cross-seed agreement and biology is tested afterwards, so the aging program passing at
r=0.982 is a consequence of the ordering, not the reason for it; and (b) the separation is
unambiguous, 0.978-0.998 against six programs below 0.6 with nothing sitting at the boundary.

It is NOT a way to choose K. Effective K comes out 8-13 whether we ask for 10, 20 or 30, so the
sweep stays a robustness check; the replication count MEASURES effective K rather than selecting it.

**Cross-K check, still to do.** Hungarian-match programs across K as well as across seeds. If the
same 8 programs appear at K=10 and K=30, the model is finding structure that survives changing the
rank -- a much stronger claim. If each K finds its own set, then the 8 are stable GIVEN K=20 but not
intrinsic, which is the weaker and more careful statement we would have to make instead. Cheap to
add to the scoring pass.

## The aging result, and what is missing from it

Program 5 (K20 seed1) rises with age in every cell type tested. Confounds checked and cleared:
depth (no correlation with library size, +0.021; survives matching in every depth quintile), sex
(effect tiny and OPPOSITE; 24m is all-male yet highest), batch (single batch), dataset (tested
within TMS only -- every 2m cell is allen_brain, so 2m must be excluded from any aging claim).

    male mice only, mouse-level means:  3m [0.053 0.020 0.062 0.053 0.058]
                                        24m [0.183 0.258 0.287 0.266]
    complete separation, p = 0.0159 (the MINIMUM attainable with n=5 vs 4)

Replicates in 5 of 7 independent fits with complete separation, 6 of 7 in the right direction. It is
one of the 8 reproducible programs (r = 0.982), which is an independent check that the cross-fit
match was not luck.

**Missing:** what the program IS. Pull its high-psi junctions and see whether they form a coherent
set. Until then it is a statistical finding, not a biological one. Redo on the final models.

## Remaining checks before biology

1. apply the decision rule (above)
2. A57 indexing checks on the chosen fit -- chained as `final_score`
3. DS suite re-run: A49 phi + group, A51 (now samples phi AND has `--permute`), A52, A53, A50.
   Everything currently on file used the collapsed-phi model, including the "0.53% called on a true
   null" figure, which must NOT be quoted until re-run.
4. LeafCutter calibration re-run -- the m-dependence result (1.15 vs 3.23 at m=2, 1.93 vs 11.35 at
   m=9) is the strongest MODEL claim we have and it is from the old fit
5. A3 held-out imputation. **Bigger than it looked**: scVI and the atlas model were both trained on
   ALL cells, so a held-out evaluation needs a 70% retrain evaluated on the 30% at each downsampling
   level. A retrain, not an adapter.
6. Moran's I of phi_k over the scVI kNN graph -- fine for VISUALISATION, but the existing scVI model
   saw all cells so it cannot support a generalisation claim.

## Also requested

- run allen_brain and tabula_muris_senis separately, not just pooled
- Figure 2 rebuild: every panel in the current version validates the BetaBinomial and Binomial
  models, neither of which is shipped. Row 1 (recovery vs NMF, concentration recovery, depth) and
  row 2 (`FIG2_regime_phi_collapse.png`, built) are specified in the gameplan.

## Known open question

The regime simulation that selected the recipe was mis-calibrated: it thinned to a median of 5 reads
per ATSE where the atlas has 16, so it tested a regime harsher than reality. The atlas A/B test
between the two arms does not depend on it, but the DIAGNOSIS of why phi collapses is weaker than it
looked. `--depth-scale` has been removed; use `--tissue` to change the coverage regime instead,
since ATSE totals come verbatim from real cells.
