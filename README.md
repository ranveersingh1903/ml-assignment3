# ML Assignment 3: MLPs, CNNs, Autoencoders, SVMs and Clustering on CIFAR10

Name: Ranveer Singh · Roll no.: P25CS0007 · Split seed: **250007** (all digits of the roll number, in order)

Every number, table and figure in the report is generated from logged runs. Nothing is typed by hand.

## How to run (Colab, free GPU)

1. Upload `ml-a3.zip` to the top level of your Google Drive (or push this folder to GitHub).
2. Open `run_colab.ipynb` in Colab (in Drive: right-click → Open with → Google Colaboratory).
3. Runtime → Change runtime type → **T4 GPU**, then Runtime → **Run all**.
4. Wait about 3–4 h. If Colab disconnects, run all again: finished models are kept in Drive and skipped.
5. Download `MyDrive/a3_results.zip`. It contains `results/` (logs, per-epoch histories, summaries), `figures/`, `report/` (with `main.pdf` if you ran the LaTeX cell) and `splits/`.

Locally or on any GPU machine:
```bash
pip install -r requirements.txt
bash scripts/run_all.sh --smoke     # ~1 min on CPU: synthetic data, checks the whole pipeline
bash scripts/run_all.sh             # full run: downloads CIFAR10, trains everything, builds report/main.pdf
```

## Layout

| Path | What |
|---|---|
| `configs/default.yaml` | every hyper-parameter, budget, grid and seed |
| `splits/cifar10_split.npz` | saved 45k/5k stratified split indices (created on first run) |
| `splits/q4_subset.npz` | fixed stratified 10k subset for Q4's SVMs, drawn before model selection |
| `src/common.py` | data, split, the **single** training loop used by every classifier, metrics |
| `src/models.py` | MLP, Patch-MLP, CNN, ConvAE, parameter-budget matching |
| `src/transforms.py` | pixel permutation (Q2b), per-image block shuffle (Q2c), Gaussian noise (Q3b) |
| `src/q1.py` … `src/q4.py` | one script per question |
| `src/make_report_assets.py` | `results/*.json` → `figures/*.pdf`, `report/generated/*.tex` |
| `results/runs/*.json` | per-epoch logs of every trained model |
| `results/q*.json` | per-question summaries |
| `report/main.tex` | CVPR-format report |
| `PREDICTIONS.md` | pre-registered predictions, fairness criteria and decision rules |
| `src/conclusions.py` | applies those decision rules to the results and writes the conclusions |
| `run_colab.ipynb` | one-click Colab runner |

## Reuse across questions
- **Q1(a) MLP** (`q1_mlp_*`) and its pixel-permuted runs (`q1_perm_mlp_*`) are reused as the Q2 MLP.
- **Q2(a) CNN, seed 0** (`q2_cnn_lr*_s0`) is the fixed classifier in Q3(b).
- **Q3 encoders** (`results/features/*.npz`) provide the Q4 representation. It is chosen automatically by the best linear-probe *validation* accuracy.

## Before you submit
1. **Commit `PREDICTIONS.md` before the full run** (it fixes the fairness criteria, predictions and decision rules).
2. Put your GitHub URL in `configs/default.yaml` (`repo_url`) before running.
3. Conclusions are written automatically by the pre-registered rules in `src/conclusions.py`. Read the report once anyway: you must be able to explain every sentence.
4. For the exact CVPR look, copy `cvpr.sty` from the official CVPR author kit into `report/` (picked up automatically).
5. Commit everything (except `results/checkpoints/` and `data/`), then `python src/stamp_commit.py`, recompile `report/main.tex`, and submit the PDF.

See `DELIVERABLES.md` for where each required item appears.
