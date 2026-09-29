"""pm.workspace: the generated uv-workspace root for plugin deps.

The workspace root is a pm-GENERATED project (never the committed
pyproject.toml — sealed installs are read-only and member lists are
machine-specific). Its pyproject = core's pyproject verbatim +
``[tool.uv.workspace] members`` pointing at each snapshotted plugin.
``uv lock`` unions core + plugin deps into ONE lock; conflict = loud refusal.
"""

from __future__ import annotations

import os
import subprocess
import shutil
import sys
from pathlib import Path

import pytest

import pm.workspace as ws
from pm.environment import managed_environment
from tests.pm.test_environment_build import locked_project  # noqa: F401




@pytest.fixture
def layout(locked_project, tmp_path, monkeypatch):
    core, uv, _ = locked_project
    manifest = core / "pyproject.toml"
    manifest.write_text(manifest.read_text().replace('[tool.uv.workspace]\nmembers=["member"]\n', ""))
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (uv, Path(sys.executable)))
    monkeypatch.setattr(ws.paths, "repo_root", lambda: core)
    return tmp_path, core, core / "member", tmp_path / "store"




def test_preparation_refuses_existing_workspace_without_mutating_it(layout):
    from pm.package import InstallError

    tmp, core, plug_a, _ = layout
    root = tmp / "workspace"
    environment = managed_environment(tmp / "env")
    kwargs = dict(root=root, source=core, seed_lock=None,
                  environment=environment)
    ws.lock_and_sync([plug_a], [], **kwargs)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    (plug_a / "pyproject.toml").write_text('changed after publication')
    with pytest.raises(InstallError, match="fresh"):
        ws.lock_and_sync([], [], **kwargs)
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()} == before


def test_missing_explicit_seed_cannot_silently_resolve_new_versions(layout):
    tmp, core, plug_a, _ = layout
    with pytest.raises(FileNotFoundError):
        ws.lock_and_sync([plug_a], [], root=tmp / "workspace", source=core,
                         seed_lock=tmp / "missing.lock", environment=managed_environment(tmp / "env"))
    assert not (tmp / "env").exists()


def test_core_quarantine_covers_core_packages_and_not_plugin_ones(layout, locked_project):
    """Regression for #120076: Hermes's 14-day cutoff must not filter a plugin's own deps.

    A global ``exclude-newer`` in the generated root made a catalog pin floored on a fresh
    release unresolvable. The cutoff now travels per package: every registry package in
    core's lock keeps it (or core's explicit exemption), plugin-only packages follow the
    plugin's policy, and the rewritten root still locks and syncs.
    """
    import tomllib

    tmp, core, plug_a, _ = layout
    _, uv, env = locked_project
    manifest = core / "pyproject.toml"
    manifest.write_text(manifest.read_text().replace("[tool.uv]\n", '[tool.uv]\nexclude-newer="14 days"\n', 1)
                        + '[tool.uv.exclude-newer-package]\nBase_Dep = false\n')
    subprocess.run([str(uv), "lock", "--python", sys.executable], cwd=core, env=env, check=True,
                   capture_output=True)
    core_packages = {p["name"] for p in tomllib.loads((core / "uv.lock").read_text())["package"]
                     if "registry" in p.get("source", {})}
    assert {"base-dep", "chosen-dep"} <= core_packages and "member-dep" not in core_packages

    root = tmp / "workspace"
    ws.lock_and_sync([plug_a], [], root=root, source=core, seed_lock=core / "uv.lock",
                     environment=managed_environment(tmp / "env"))

    policy = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["uv"]
    assert "exclude-newer" not in policy
    per_package = policy["exclude-newer-package"]
    assert per_package == {name: False if name == "base-dep" else "14 days" for name in core_packages}
    locked = {p["name"] for p in tomllib.loads((root / "uv.lock").read_text())["package"]}
    assert "member-dep" in locked and "member-dep" not in per_package






# --- classified failures + staging surface (FINAL-RUNTIME-CONTRACT) ---


def test_classify_network_failure_stays_generic():
    from pm.workspace import ResolutionConflict, classify_uv_failure

    err = classify_uv_failure("lock", 1, "error: Failed to fetch https://pypi.org (timed out)")
    assert not isinstance(err, ResolutionConflict)


def test_sync_failure_is_never_a_conflict(layout, monkeypatch):
    from pm.package import InstallError

    tmp, core, _, _ = layout
    environment = managed_environment(tmp / "candidate")
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kwargs:
                        subprocess.CompletedProcess(cmd, 1, "", "Failed to download wheel"))
    with pytest.raises(InstallError) as excinfo:
        ws.lock_and_sync([], [], root=tmp / "workspace", source=core,
                         seed_lock=None, environment=environment, frozen=True)
    assert not isinstance(excinfo.value, ws.ResolutionConflict)


def test_staging_root_and_env_are_honored_without_live_mutation(layout, monkeypatch):
    tmp, core, _, _ = layout
    staging = tmp / "staging-ws"
    monkeypatch.setenv("PM_WORKSPACE_TEST_SENTINEL", "live")
    environment = managed_environment(tmp / "staging-venv", env={
        "PATH": "/staged/bin", "PM_WORKSPACE_TEST_SENTINEL": "staged",
    })
    # The prepared environment is authoritative; workspace never discovers tools.
    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: pytest.fail("PATH discovery"))
    seen = []
    def run(cmd, **kwargs):
        seen.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", run)
    ws.lock_and_sync([], [], root=staging, source=core, seed_lock=None, environment=environment)
    assert [cmd[1] for cmd, _ in seen] == ["lock", "sync"]
    for cmd, kwargs in seen:
        assert Path(cmd[0]) == environment.uv
        assert Path(kwargs["cwd"]) == staging
        assert kwargs["env"]["PM_WORKSPACE_TEST_SENTINEL"] == "staged"
        assert kwargs["env"]["UV_CACHE_DIR"] == str(environment.cache)
        assert kwargs["env"]["UV_PROJECT_ENVIRONMENT"] == str(environment.destination)
        assert kwargs["env"]["UV_PYTHON"] == str(environment.python)
    assert os.environ["PM_WORKSPACE_TEST_SENTINEL"] == "live"


@pytest.mark.parametrize("runtime_conflict", [False, True])
def test_plugin_development_dependencies_are_separate_environments(tmp_path, runtime_conflict):
    """Real uv resolution keeps plugin dev pins without weakening runtime constraints."""
    import json
    import tomllib

    from pm.environment import PythonEnvironment
    from tests.pm import _fixtures

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    for version in ("1.0", "2.0", "3.0"):
        _fixtures._wheel(wheels, "devdep", version)
        _fixtures._wheel(wheels, "runtimedep", version)
    core = tmp_path / "core"
    core.mkdir()
    (core / "pyproject.toml").write_text(
        '[project]\nname="core"\nversion="1"\nrequires-python=">=3.11"\n'
        'dependencies=["runtimedep==1.0"]\n'
        '[project.optional-dependencies]\none=[]\ntwo=[]\n'
        '[dependency-groups]\ndev=["devdep==1.0"]\n'
        '[tool.uv]\npackage=false\nno-index=true\ndefault-groups=[]\n'
        'conflicts=[[{package="core",extra="one"},{package="core",extra="two"}]]\n'
        f'find-links=[{json.dumps(wheels.as_posix())}]\n', encoding="utf-8",
    )
    plugins = []
    for index, version in enumerate(("2.0", "3.0")):
        plugin = tmp_path / f"plugin-{index}"
        plugin.mkdir()
        runtime = "2.0" if runtime_conflict else "1.0"
        (plugin / "pyproject.toml").write_text(
            f'[project]\nname="plugin-{index}"\nversion="1"\nrequires-python=">=3.11"\n'
            f'dependencies=["runtimedep=={runtime}"]\n'
            f'[project.optional-dependencies]\ndev=["devdep=={version}"]\nfeature=[]\n'
            '[tool.uv]\npackage=false\n', encoding="utf-8",
        )
        plugins.append(plugin)
    before = {p / "pyproject.toml": (p / "pyproject.toml").read_bytes() for p in [core, *plugins]}
    uv = shutil.which("uv")
    assert uv is not None, "real resolver is required"
    environment = PythonEnvironment(
        uv=Path(uv), python=Path(sys.executable), destination=tmp_path / "env",
        cache=tmp_path / "cache", env=dict(os.environ), offline=True,
    )
    root = tmp_path / "snapshot"
    if runtime_conflict:
        with pytest.raises(ws.ResolutionConflict):
            ws.lock_and_sync(plugins, [], root=root, source=core, seed_lock=None, environment=environment)
    else:
        ws.lock_and_sync(plugins, [], root=root, source=core, seed_lock=None, environment=environment)
        document = tomllib.loads((root / "pyproject.toml").read_text())
        assert document["tool"]["uv"]["conflicts"][0] == [
            {"package": "core", "extra": "one"}, {"package": "core", "extra": "two"},
        ]
        for relative, original in zip(document["tool"]["uv"]["workspace"]["members"], plugins):
            generated = tomllib.loads((root / relative / "pyproject.toml").read_text())
            original_doc = tomllib.loads((original / "pyproject.toml").read_text())
            assert generated["project"]["optional-dependencies"] == original_doc["project"]["optional-dependencies"]
        probe = subprocess.run(
            [str(environment.executable), "-I", "-c",
             "import runtimedep; import importlib.util; "
             "assert runtimedep.__version__ == '1.0'; assert importlib.util.find_spec('devdep') is None"],
            capture_output=True, text=True, timeout=30,
        )
        assert probe.returncode == 0, probe.stderr
        selected = environment._run(
            ["sync", "--frozen", "--all-packages", "--group", "dev", "--extra", "dev"],
            cwd=root, timeout=30,
        )
        assert selected.returncode != 0
        assert "conflict" in selected.stderr.lower(), selected.stderr
    assert all(p.read_bytes() == body for p, body in before.items())
