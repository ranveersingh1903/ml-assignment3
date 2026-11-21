#!/usr/bin/env bash
# Reproduces every number in the report.  Usage:
#   bash scripts/run_all.sh            # full run (GPU strongly recommended, ~3-4 h on a T4)
#   bash scripts/run_all.sh --smoke    # 1-minute pipeline check on synthetic data (CPU ok)
# Runs are cached in results/; re-running resumes where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."
FLAG="${1:-}"
LOG=results${FLAG:+_smoke}/logs; mkdir -p "$LOG"
cd src
python q1.py $FLAG 2>&1 | tee "../$LOG/q1.log"
python q2.py $FLAG 2>&1 | tee "../$LOG/q2.log"
python q3.py $FLAG 2>&1 | tee "../$LOG/q3.log"   # needs q2 (fixed classifier)
python q4.py $FLAG 2>&1 | tee "../$LOG/q4.log"   # needs q3 (representation)
python make_report_assets.py $FLAG 2>&1 | tee "../$LOG/assets.log"
cd ../report
if command -v pdflatex >/dev/null; then
  if [ -n "$FLAG" ]; then pdflatex -interaction=nonstopmode -jobname=main_smoke "\def\smoke{1}\input{main}" >/dev/null; pdflatex -interaction=nonstopmode -jobname=main_smoke "\def\smoke{1}\input{main}" >/dev/null
  else pdflatex -interaction=nonstopmode main >/dev/null; pdflatex -interaction=nonstopmode main >/dev/null; fi
  echo "report compiled in report/"
fi
