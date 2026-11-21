"""Writes the current git commit hash into report/generated/numbers.tex without re-running anything.
Run after your final commit:  python src/stamp_commit.py  (then recompile the report)."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
h = subprocess.run(["git", "rev-parse", "--short=10", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
f = ROOT / "report" / "generated" / "numbers.tex"
s = re.sub(r"\\newcommand\{\\CommitHash\}\{[^}]*\}", lambda m: "\\newcommand{\\CommitHash}{" + (h or "UNCOMMITTED") + "}",
           f.read_text())
f.write_text(s)
print("CommitHash ->", h)
