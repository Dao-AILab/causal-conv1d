import ast
import os
from pathlib import Path
import shutil
import socket
from types import SimpleNamespace
from unittest.mock import Mock
import urllib.error
import urllib.request

import pytest
from setuptools import Distribution
from wheel.bdist_wheel import bdist_wheel


@pytest.fixture
def cached_wheels(tmp_path, monkeypatch):
    # Load the real command without setup.py's top-level CUDA probing or setup().
    setup_path = Path(__file__).resolve().parents[1] / "setup.py"
    tree = ast.parse(setup_path.read_text(encoding="utf-8"), filename=str(setup_path))
    command = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CachedWheelsCommand"
    )
    wheel_filename = tmp_path / "cached.whl"
    wheel_url = "https://example.invalid/cached.whl"
    namespace = {
        "_bdist_wheel": bdist_wheel,
        "FORCE_BUILD": False,
        "get_wheel_url": Mock(return_value=(wheel_url, str(wheel_filename))),
        "os": os,
        "shutil": shutil,
        "urllib": urllib,
    }
    exec(
        compile(ast.Module(body=[command], type_ignores=[]), str(setup_path), "exec"),
        namespace,
    )
    cmd = namespace["CachedWheelsCommand"](
        Distribution({"name": "causal_conv1d", "version": "1.7.0"})
    )
    cmd.dist_dir = str(tmp_path / "dist")
    cmd.get_tag = Mock(return_value=("cp313", "cp313", "linux_x86_64"))
    download = Mock()
    build = Mock()
    monkeypatch.setattr(urllib.request, "urlretrieve", download)
    monkeypatch.setattr(bdist_wheel, "run", build)
    return SimpleNamespace(
        cmd=cmd,
        namespace=namespace,
        download=download,
        build=build,
        filename=wheel_filename,
        url=wheel_url,
    )


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError(
            socket.gaierror(-3, "Temporary failure in name resolution")
        ),
        urllib.error.URLError("TLS handshake failed"),
        urllib.error.HTTPError(
            "https://example.invalid/cached.whl", 404, "Not Found", {}, None
        ),
        urllib.error.ContentTooShortError("incomplete download", ("cached.whl", None)),
    ],
    ids=["dns", "tls", "http-404", "short-download"],
)
def test_download_error_builds_from_source(cached_wheels, error, capsys):
    cached_wheels.download.side_effect = error

    cached_wheels.cmd.run()

    cached_wheels.download.assert_called_once_with(
        cached_wheels.url, str(cached_wheels.filename)
    )
    cached_wheels.build.assert_called_once_with()
    cached_wheels.cmd.get_tag.assert_not_called()
    assert not Path(cached_wheels.cmd.dist_dir).exists()
    assert "Building from source" in capsys.readouterr().out


def test_force_build_skips_wheel_lookup(cached_wheels):
    cached_wheels.namespace["FORCE_BUILD"] = True

    cached_wheels.cmd.run()

    cached_wheels.namespace["get_wheel_url"].assert_not_called()
    cached_wheels.download.assert_not_called()
    cached_wheels.build.assert_called_once_with()


def test_successful_download_moves_wheel_without_building(cached_wheels):
    cached_wheels.download.side_effect = lambda url, filename: Path(
        filename
    ).write_bytes(b"wheel contents")

    cached_wheels.cmd.run()

    cached_wheels.download.assert_called_once_with(
        cached_wheels.url, str(cached_wheels.filename)
    )
    cached_wheels.build.assert_not_called()
    assert not cached_wheels.filename.exists()
    wheels = list(Path(cached_wheels.cmd.dist_dir).glob("*.whl"))
    assert [wheel.name for wheel in wheels] == [
        "causal_conv1d-1.7.0-cp313-cp313-linux_x86_64.whl"
    ]
    assert wheels[0].read_bytes() == b"wheel contents"


@pytest.mark.parametrize(
    "error", [PermissionError("download denied"), ValueError("bad URL")]
)
def test_unrelated_download_errors_propagate(cached_wheels, error):
    cached_wheels.download.side_effect = error

    with pytest.raises(type(error)) as exc_info:
        cached_wheels.cmd.run()

    assert exc_info.value is error
    cached_wheels.build.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        PermissionError("archive destination denied"),
        urllib.error.URLError("archive error"),
    ],
    ids=["permission", "url-error-outside-download"],
)
def test_archive_move_error_propagates(cached_wheels, monkeypatch, error):
    monkeypatch.setattr(shutil, "move", Mock(side_effect=error))

    with pytest.raises(type(error)) as exc_info:
        cached_wheels.cmd.run()

    assert exc_info.value is error
    cached_wheels.build.assert_not_called()


def test_source_build_error_propagates(cached_wheels):
    cached_wheels.download.side_effect = urllib.error.URLError("offline")
    error = RuntimeError("source compilation failed")
    cached_wheels.build.side_effect = error

    with pytest.raises(RuntimeError) as exc_info:
        cached_wheels.cmd.run()

    assert exc_info.value is error
    cached_wheels.build.assert_called_once_with()
