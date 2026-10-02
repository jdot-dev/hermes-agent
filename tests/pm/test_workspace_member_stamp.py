"""A member fingerprint covers exactly the bytes copied into its workspace snapshot."""

import subprocess

from pm.workspace import _workspace_member, members_stamp


def test_member_stamp_tracks_gitignored_files_copied_into_snapshot(tmp_path):
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    subprocess.run(["git", "init", "--quiet", str(plugin)], check=True)
    (plugin / "pyproject.toml").write_text(
        '[project]\nname="stamp-plugin"\nversion="1"\n'
        '[tool.uv]\npackage=true\n'
    )
    (plugin / "plugin.yaml").write_text("name: stamp-plugin\n")
    (plugin / ".gitignore").write_text("state.json\nconfig.yaml\ndashboard/dist/\n")
    state = plugin / "state.json"
    state.write_text('{"sessions": {}}\n')
    (plugin / "config.yaml").write_text("enabled: true\n")
    (plugin / "dashboard" / "dist").mkdir(parents=True)
    (plugin / "dashboard" / "dist" / "index.html").write_text("packaged dashboard")

    before = members_stamp([plugin])
    copied = _workspace_member(plugin, tmp_path / "workspace", identity=plugin)

    assert (copied / "state.json").read_bytes() == state.read_bytes()
    assert (copied / "config.yaml").read_text() == "enabled: true\n"
    assert (copied / "dashboard" / "dist" / "index.html").read_text() == "packaged dashboard"
    assert members_stamp({plugin: copied}) == before

    state.write_text('{"sessions": {"live": 2}}\n')
    assert members_stamp([plugin]) != before
