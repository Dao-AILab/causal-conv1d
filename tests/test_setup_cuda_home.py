# Copyright (c) 2024, Tri Dao.

import os
import sys
import urllib.error
from unittest.mock import patch
import pytest
import torch


def test_setup_with_cuda_home_none():
    """Verify setup.py executes safely when CUDA_HOME is None without NameError or unassigned variable crashes."""
    setup_path = os.path.join(os.path.dirname(__file__), "..", "setup.py")
    with open(setup_path, "r", encoding="utf-8") as f:
        code = f.read()

    with patch("torch.utils.cpp_extension.CUDA_HOME", None), patch.object(
        sys, "argv", ["setup.py", "--name"]
    ):
        g = {"__file__": setup_path, "__name__": "__main__", "sys": sys}
        exec(code, g)

        assert g["ext_modules"] == []
        assert "bdist_wheel" in g["cmdclass"]
        assert "build_ext" not in g["cmdclass"]


def test_setup_cpu_torch_clear_error():
    """Verify get_wheel_url provides a clear error message when PyTorch has no CUDA support."""
    setup_path = os.path.join(os.path.dirname(__file__), "..", "setup.py")
    with open(setup_path, "r", encoding="utf-8") as f:
        code = f.read()

    with patch.object(torch.version, "cuda", None), patch.object(
        sys, "argv", ["setup.py", "--name"]
    ):
        g = {"__file__": setup_path, "__name__": "__main__", "sys": sys}
        exec(code, g)

        with pytest.raises(RuntimeError, match="PyTorch was built without CUDA support"):
            g["get_wheel_url"]()


def test_cached_wheels_command_force_build_without_compiler():
    """Verify FORCE_BUILD fails with an informative error if no compiler is available."""
    setup_path = os.path.join(os.path.dirname(__file__), "..", "setup.py")
    with open(setup_path, "r", encoding="utf-8") as f:
        code = f.read()

    with patch("torch.utils.cpp_extension.CUDA_HOME", None), patch.dict(
        os.environ, {"CAUSAL_CONV1D_FORCE_BUILD": "TRUE"}
    ), patch.object(sys, "argv", ["setup.py", "--name"]):
        g = {"__file__": setup_path, "__name__": "__main__", "sys": sys}
        exec(code, g)

        from setuptools.dist import Distribution

        dist = Distribution({"name": "causal_conv1d"})
        cmd = g["cmdclass"]["bdist_wheel"](dist)

        with pytest.raises(RuntimeError, match="no compiler was found"):
            cmd.run()


def test_cached_wheels_command_missing_wheel_without_compiler():
    """Verify missing precompiled wheel fails with an informative error if no compiler is available."""
    setup_path = os.path.join(os.path.dirname(__file__), "..", "setup.py")
    with open(setup_path, "r", encoding="utf-8") as f:
        code = f.read()

    with patch("torch.utils.cpp_extension.CUDA_HOME", None), patch.object(
        sys, "argv", ["setup.py", "--name"]
    ), patch(
        "urllib.request.urlretrieve",
        side_effect=urllib.error.HTTPError("url", 404, "Not Found", {}, None),
    ):
        g = {"__file__": setup_path, "__name__": "__main__", "sys": sys}
        exec(code, g)

        from setuptools.dist import Distribution

        dist = Distribution({"name": "causal_conv1d"})
        cmd = g["cmdclass"]["bdist_wheel"](dist)

        with pytest.raises(RuntimeError, match="cannot build from source because no compiler was found"):
            cmd.run()
