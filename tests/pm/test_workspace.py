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
from tests.pm.test_environment_build import locked_project




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


def test_plugin_workspace_trusts_core_lock_instead_of_rechecking_release_age(layout, locked_project):  # health: allow F811 -- pytest injects the imported locked_project fixture by parameter name
    """Release age is checked when Hermes locks, never on the user's machine (#135589).

    The generated root carries no ``exclude-newer`` (a mirror without upload times makes
    uv reject every file it cannot date) and adds no constraints of its own: core's ranges
    still bound what a plugin may move, and plugin-only packages stay free (#120076).
    """
    import tomllib

    tmp, core, plug_a, _ = layout
    _, uv, env = locked_project
    manifest = core / "pyproject.toml"
    manifest.write_text(manifest.read_text().replace("[tool.uv]\n", '[tool.uv]\nexclude-newer="3 days"\n', 1)
                        + '[tool.uv.exclude-newer-package]\nBase_Dep = false\n')
    subprocess.run([str(uv), "lock", "--python", sys.executable], cwd=core, env=env, check=True,
                   capture_output=True)
    core_lock = tomllib.loads((core / "uv.lock").read_text())["package"]
    core_packages = {p["name"]: p["version"] for p in core_lock if "registry" in p.get("source", {})}
    assert {"base-dep", "chosen-dep"} <= set(core_packages) and "member-dep" not in core_packages

    root = tmp / "workspace"
    ws.lock_and_sync([plug_a], [], root=root, source=core, seed_lock=core / "uv.lock",
                     environment=managed_environment(tmp / "env"))

    policy = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["uv"]
    assert "exclude-newer" not in policy and "exclude-newer-package" not in policy
    assert "constraint-dependencies" not in policy
    locked = {p["name"] for p in tomllib.loads((root / "uv.lock").read_text())["package"]}
    assert "member-dep" in locked






# --- classified failures + staging surface (FINAL-RUNTIME-CONTRACT) ---


def test_classify_network_failure_stays_generic():
    from pm.workspace import ResolutionConflict, classify_uv_failure

    err = classify_uv_failure("lock", 1, "error: Failed to fetch https://pypi.org (timed out)")
    assert not isinstance(err, ResolutionConflict)


def test_core_only_generation_carries_no_release_age_cutoff(layout):
    """A core-only install (hermes update) is a frozen sync: the cutoff there only gates the build
    tools uv fetches, and a mirror without upload times rejects every one of them (#135589)."""
    tmp, core, _, _ = layout
    manifest = core / "pyproject.toml"
    manifest.write_text(manifest.read_text().replace("[tool.uv]\n", '[tool.uv]\nexclude-newer = "3 days"\n', 1))
    root = tmp / "workspace"
    ws._generate_pyproject([], root, source=core)
    generated = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "exclude-newer" not in generated and "[tool.uv]" in generated


def test_undated_index_failure_names_the_index_not_a_conflict():
    """uv's "has no publish time" is the index omitting upload times (#135589), not a conflict."""
    from pm.workspace import ResolutionConflict, classify_uv_failure

    seeded = ("  x No solution found when resolving dependencies:\n"
              "  Because resvg-py==0.4.0 has no publish time and hermes-agent depends on resvg-py==0.4.0, "
              "we can conclude that hermes-agent's requirements are unsatisfiable.")
    cold = ("warning: cffi-2.0.0.tar.gz is missing an upload date, but user provided: 2026-09-25T00:00:00Z\n"
            "  x No solution found when resolving dependencies:\n"
            "  Because there are no versions of cffi and you require cffi>=2.0.0, we can conclude that "
            "your requirements are unsatisfiable.\n"
            "hint: `cffi` was filtered by `exclude-newer` to only include packages uploaded before 2026-09-25")
    for output in (seeded, cold):
        err = classify_uv_failure("lock", 1, output)
        assert not isinstance(err, ResolutionConflict)
        assert "upload times" in err.remedy and "pypi.org" in err.remedy
    # A real conflict on a partly undated mirror still reads as a conflict.
    conflict = cold.splitlines()[0] + "\n  x No solution found: Because a==1 depends on b<2 and c==1 depends on b>=2"
    assert isinstance(classify_uv_failure("lock", 1, conflict), ResolutionConflict)


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
@pytest.mark.parametrize("development_extra", ["dev", "test"])
def test_plugin_development_dependencies_are_separate_environments(tmp_path, runtime_conflict, development_extra):
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
        '[dependency-groups]\ndev=["devdep==1.0"]\ntest=["devdep==1.0"]\n'
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
            f'[project.optional-dependencies]\n{development_extra}=["devdep=={version}"]\nfeature=[]\n'
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
        own_development = environment._run(
            ["sync", "--frozen", "--all-packages", "--group", "dev", "--group", "test"],
            cwd=root, timeout=30,
        )
        assert own_development.returncode == 0, own_development.stderr
        selected = environment._run(
            ["sync", "--frozen", "--all-packages", "--group", "dev", "--extra", development_extra],
            cwd=root, timeout=30,
        )
        assert selected.returncode != 0
        assert "conflict" in selected.stderr.lower(), selected.stderr
    assert all(p.read_bytes() == body for p, body in before.items())
