# Pre-registered predictions and fairness criteria

Commit this file **before** running `scripts/run_all.sh`; the git history then shows the
predictions were fixed before any result existed (Q2(b) explicitly asks for this).

## Fairness criterion (Q1(a), Q2(a))
Compared classifiers: trainable parameters within ±2% of `q1.param_budget`; identical
data, preprocessing and split; no augmentation; AdamW, wd 5e-4, batch 256, cosine
schedule, same number of epochs; same LR grid searched at seed 0 and selected on
validation; same 3 seeds; checkpoint = best validation accuracy. FLOPs are reported,
not matched.

## Q1(a)
- H_bias (chosen): MLP training accuracy is much higher than validation; widening it raises
  training accuracy far more than validation; the equal-budget Patch-MLP beats the MLP
  clearly; under pixel permutation the Patch-MLP advantage disappears and the MLP is unchanged.
- H_cap would predict low training accuracy and validation accuracy rising steadily with width.
- H_opt would predict that the equal-budget deep MLP closes most of the gap.

## Q1(b)
Validation accuracy peaks while the noisy-label fit is still low and most noisy examples are
predicted as their *true* class; the noisy fit then climbs towards 100% while validation
accuracy and true-label agreement on the noisy subset fall.

## Q2(b) — fixed pixel permutation
MLP test accuracy unchanged within seed noise (about ±1 point); CNN drops sharply, to about
the MLP level or below.

## Q2(c) — per-image block shuffle (16/8/4 px)
- E_local (texture / bag of local features): small drop at 16 px, moderate at 8 px, large at 4 px.
- E_global (shape / layout): large drop already at 16 px.
Our expectation: E_local dominates, with shape-heavy classes (e.g. horse, deer, cat/dog)
losing more than texture/colour-heavy classes (e.g. ship, airplane, frog).

## Q3
- (a) The largest bottleneck has the lowest validation MSE.
- (b) The DAE recovers part, but not all, of the clean-vs-noisy accuracy gap; it is capped by
  the blur of the AE itself (the "AE on clean" row).
- (c) Reconstruction quality and linear-probe accuracy do not rank the encoders identically.

## Decision rules (applied automatically by `src/conclusions.py`)
All differences are in percentage points (pp) of test accuracy, averaged over 3 seeds.

| Claim | Supported if |
|---|---|
| Q1(a) reject H_cap | MLP train-val gap at the selected epoch >= 10 pp **and** the width sweep raises val by < half of what it raises train |
| Q1(a) reject H_opt | equal-budget deep MLP gains < 2 pp over the MLP |
| Q1(a) support H_bias | Patch-MLP gains >= max(3 pp, 2 sd) **and** permutation removes >= half of that gain **and** the MLP moves < 1.5 pp under permutation |
| Q1(b) structure before memorisation | at the val peak, true-label agreement on noisy examples > noisy-label fit **and** the noisy fit rises >= 20 pp afterwards |
| Q2(b) supports Q2(a) | MLP moves < 1.5 pp **and** the CNN keeps < 25% of its clean advantage |
| Q2(c) E_local / E_global | drop at 16 px < 40% / >= 60% of the drop at 4 px (otherwise both) |
| Q3(b) DAE recovers the needed information | closes >= 50% of the clean-noisy gap **and** is within 3 pp of the "clean AE on clean" ceiling |
| Q3(c) reconstruction and class info agree | ranking the three AEs by test MSE = ranking by linear-probe accuracy, and the DAE does not reverse it |
