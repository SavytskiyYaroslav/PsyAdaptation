import json
import os
import subprocess
import sys
from pathlib import Path


def _cli_env() -> dict:
    """Ensure `python -m psyadapt` works even without `pip install -e .`.

    CI should still install the package, but this makes local/isolated runs robust.
    """
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[1]
    src = str(repo_root / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def test_cli_outputs_json(tmp_path):
    data = "a,b\n1,0\n1,0\n0,1\n1,0\n"
    p1 = tmp_path / "e1.csv"
    p2 = tmp_path / "e2.csv"
    p1.write_text(data, encoding="utf-8")
    p2.write_text(data, encoding="utf-8")

    # Call as module: python -m psyadapt ...
    cmd = [sys.executable, "-m", "psyadapt", str(p1), str(p2), "--pretty"]
    out = subprocess.check_output(cmd, text=True, env=_cli_env())

    rep = json.loads(out)
    assert rep["inputs"]["n_files"] == 2
    assert "results" in rep
    assert rep["results"]["icc"]["estimator"] == "ICC"

def test_cli_forced_krippendorff(tmp_path):
    data = "a,b\n1,0\n1,0\n0,1\n1,0\n"
    p1 = tmp_path / "e1.csv"
    p2 = tmp_path / "e2.csv"
    p3 = tmp_path / "e3.csv"
    for p in (p1, p2, p3):
        p.write_text(data, encoding="utf-8")

    cmd = [
        sys.executable, "-m", "psyadapt",
        str(p1), str(p2), str(p3),
        "--primary", "krippendorff",
        "--pretty",
    ]
    out = subprocess.check_output(cmd, text=True, env=_cli_env())
    rep = json.loads(out)

    assert rep["summary"]["primary_estimator"] == "KrippendorffAlpha"

def test_cli_forced_krippendorff_invalid(tmp_path):
    data = "a,b\n1,0\n1,0\n0,1\n1,0\n"
    p1 = tmp_path / "e1.csv"
    p1.write_text(data, encoding="utf-8")

    cmd = [
        sys.executable, "-m", "psyadapt",
        str(p1),
        "--primary", "krippendorff",
        "--pretty",
    ]

    proc = subprocess.run(cmd, text=True, capture_output=True, env=_cli_env())
    assert proc.returncode == 0

    rep = json.loads(proc.stdout)
    assert rep["status"] == "error"
    assert any("primary" in (e.get("code") or "") or "incompatible" in (e.get("code") or "")
               for e in rep.get("errors", [])) or len(rep.get("errors", [])) > 0

