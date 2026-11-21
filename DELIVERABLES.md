# Deliverables checklist: where each required item appears in the report

| Required by the assignment | Where |
|---|---|
| **Common** split: 45k/5k stratified, roll-number seed, saved | `splits/cifar10_split.npz`; Sec. 1 |
| Validation-only selection; test once at the end | `src/common.py` (`best_val_acc`); every script |
| Name, roll number, repo link, final commit hash on page 1 | title block (`configs/default.yaml` + `src/stamp_commit.py`) |
| Reuse identified, with the question that introduced it | Sec. 1 "Reuse"; Q2 table ("reused Q1"); Q3(b) text; Q4 text |
| **Q1** model spec + parameter count for each comparison | Table `tab:q1spec`, `tab:q1a` |
| Q1 train and validation curves | Fig. `q1a_curves.pdf`; `results/runs/q1_*.json` |
| Q1 one plot/table for memorisation | Fig. `q1b_memorisation.pdf` + Table `tab:q1b` |
| Q1 final test accuracy of the comparison models | Table `tab:q1a` (mean ± sd, 3 seeds) |
| Q1 interpretation | `\ConcQoneA`, `\ConcQoneB` (pre-registered rules) |
| **Q2** fairness defined before results | `PREDICTIONS.md` (commit before running); Sec. 3 |
| Q2 one table with models and parameter counts | Table `tab:q2` |
| Q2 test results: clean, spatially disturbed, third condition | Table `tab:q2` |
| Q2 expected result stated beforehand | `PREDICTIONS.md`; "Prediction" sentences in Sec. 3 |
| Q2 test confidence / error analysis | Table `tab:q2conf`; Fig. `q2_analysis.pdf`; per-class and confusion text |
| Q2 same transformation on train and test | `src/transforms.py`; applied to train/val/test in `src/q2.py` |
| Q2 short conclusion linking the three experiments | `\ConcQtwoLink` |
| **Q3** three bottleneck sizes, params, val reconstruction error | Table `tab:q3a` |
| Q3 test reconstruction error of the selected AE | caption of `tab:q3a` |
| Q3 corruption and denoising protocol | Sec. 4(b) |
| Q3 test classification: corrupted, clean-AE, DAE reconstructions | Table `tab:q3b` |
| Q3 test comparison: reconstruction vs class information | Table `tab:q3c`; Fig. `q3.pdf` |
| **Q4** fixed subset chosen before selection, reported | `splits/q4_subset.npz`; Sec. 5 |
| Q4 full C × kernel table (train, val, SV fraction) | Table `tab:q4a` (6 C values over 5 orders × 3 kernels incl. RBF) |
| Q4 plot of val acc and SV fraction vs C per kernel | Fig. `q4a.pdf` |
| Q4 final test accuracy of the selected SVM | caption of `tab:q4a` |
| Q4 SVs + near-boundary test images with scores, ≥3 pairs | Fig. `q4b_pairs.pdf`; Table `tab:q4b` |
| Q4 test ARI, NMI, contingency table | Fig. `q4c_contingency.pdf`; Sec. 5(c) |
| Q4 quantitative classification-vs-clustering disagreement | `\ConcQfourC`; Table `tab:q4cpc` |
| **Repo** code, split, configs, final logs, figure scripts, env file, commands | `src/`, `splits/`, `configs/`, `results/logs/`, `src/make_report_assets.py`, `environment.yml`, `README.md` |
