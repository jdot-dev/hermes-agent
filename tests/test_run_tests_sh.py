from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_TESTS = REPO_ROOT / "scripts" / "run_tests.sh"


def _write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_runner_skips_local_venv_without_pytest(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(RUN_TESTS, scripts / "run_tests.sh")
    (scripts / "run_tests_parallel.py").write_text("", encoding="utf-8")

    # A stale release venv may exist before the usable development venv. The
    # canonical runner must probe pytest instead of selecting on activate alone.
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "activate").write_text("", encoding="utf-8")
    _write_executable(
        tmp_path / ".venv" / "bin" / "python",
        "#!/bin/sh\nexit 99\n",
    )

    (tmp_path / "venv" / "bin").mkdir(parents=True)
    (tmp_path / "venv" / "bin" / "activate").write_text("", encoding="utf-8")
    selected = tmp_path / "selected-python"
    _write_executable(
        tmp_path / "venv" / "bin" / "python",
        "#!/bin/sh\n"
        "if [ \"$1\" = \"-c\" ]; then exit 0; fi\n"
        "if [ \"$1\" = \"-m\" ]; then exit 0; fi\n"
        f"printf selected > {selected}\n"
        "exit 0\n",
    )

    env = os.environ.copy()
    env["HOME"] = str(tmp_path / "home")
    result = subprocess.run(
        ["bash", str(scripts / "run_tests.sh"), "-q"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert selected.read_text(encoding="utf-8") == "selected"
