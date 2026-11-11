import subprocess
import sys

scripts = [
    "scripts/00_make_pseudo_raters.py",
    "scripts/10_test_cronbach.py",
    "scripts/20_test_cohen_icc.py",
    "scripts/30_test_fleiss_icc.py",
    "scripts/40_test_krippendorff_direct.py",
    "scripts/50_test_forced_primary.py",
]

for s in scripts:
    print("\n=== RUN:", s, "===")
    r = subprocess.run([sys.executable, s], check=False)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {s} (exit={r.returncode})")

print("\nALL MANUAL TESTS PASSED ✅")
