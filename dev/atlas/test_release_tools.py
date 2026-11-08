"""Regression checks for repaired-wheel and published-artifact integrity."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path

from packaging.tags import sys_tags
import pytest


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_repair_preserves_build_provenance(tmp_path):
    tool = load_tool("repair_wheel")
    native = tmp_path / "brian2_rust/_bin"
    native.mkdir(parents=True)
    for name in ("b2-runner", "b2-train"):
        (native / name).write_bytes(b"repaired ELF")
    original = {name: "old-build-hash" for name in ("b2-runner", "b2-train")}
    manifest = native / "build.json"
    manifest.write_text(json.dumps({"schema": "atlas-native-build-v1", "binaries": original,
                                    "sources": {"src/main.rs": "source-hash"}}))
    tool.refresh_native_hashes(tmp_path)
    result = json.loads(manifest.read_text())
    assert result["pre_repair_binaries"] == original
    assert result["sources"] == {"src/main.rs": "source-hash"}
    assert set(result["binaries"].values()) == {hashlib.sha256(b"repaired ELF").hexdigest()}
    previous_bytes = manifest.read_bytes()
    tool.refresh_native_hashes(tmp_path)
    assert manifest.read_bytes() == previous_bytes


def test_native_repair_rejects_missing_binary(tmp_path):
    tool = load_tool("repair_wheel")
    native = tmp_path / "brian2_rust/_bin"
    native.mkdir(parents=True)
    (native / "build.json").write_text(json.dumps({
        "schema": "atlas-native-build-v1", "binaries": {"b2-runner": "hash"},
    }))
    with pytest.raises(ValueError, match="both Linux native executables"):
        tool.refresh_native_hashes(tmp_path)


@pytest.mark.parametrize("corrupt", [None, "index", "download"])
@pytest.mark.parametrize("index", ["testpypi", "pypi"])
def test_testpypi_checks_index_and_downloaded_bytes(tmp_path, monkeypatch, corrupt, index):
    tool = load_tool("verify_testpypi")
    expected = tmp_path / "expected"
    expected.mkdir()
    filename = f"brian2_atlas-0.2.0.dev1-{next(sys_tags())}.whl"
    content = b"validated wheel content"
    (expected / filename).write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    release = {"urls": [{"filename": filename, "url": "https://example.test/wheel",
                         "digests": {"sha256": "wrong" if corrupt == "index" else digest}}]}

    def download(url, **kwargs):
        if url.endswith("/json"):
            host = "pypi.org" if index == "pypi" else "test.pypi.org"
            assert url == f"https://{host}/pypi/brian2-atlas/0.2.0.dev1/json"
            return io.BytesIO(json.dumps(release).encode())
        return io.BytesIO(b"wrong bytes" if corrupt == "download" else content)

    monkeypatch.setattr(tool.urllib.request, "urlopen", download)
    destination = tmp_path / "downloaded"
    monkeypatch.setattr(tool.sys, "argv", ["verify_testpypi.py", "0.2.0.dev1", str(expected), str(destination), "--index", index])
    if corrupt:
        with pytest.raises(ValueError, match="mismatch|differ"):
            tool.main()
        assert not (destination / filename).exists()
    else:
        tool.main()
        assert (destination / filename).read_bytes() == content


@pytest.mark.parametrize("version,ref", [
    ("0.2.0.dev1", "refs/tags/v0.2.0.dev1"),
    ("0.2.0rc1", "refs/tags/v0.2.0rc1"),
    ("0.2.0+local", "refs/tags/v0.2.0+local"),
    ("0.2.0", "refs/tags/v0.3.0"),
])
def test_production_rejects_nonrelease_tags(version, ref):
    with pytest.raises(ValueError):
        load_tool("check_release").validate_release(version, ref)


def test_production_accepts_stable_tag_and_development_dry_run():
    tool = load_tool("check_release")
    tool.validate_release("0.2.0", "refs/tags/v0.2.0")
    tool.validate_release("0.2.0.dev1", "refs/heads/dev")
