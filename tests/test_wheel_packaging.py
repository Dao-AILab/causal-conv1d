"""Exercise release wheel selection without importing or compiling CUDA."""

import ast
import platform
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.utils import parse_wheel_filename
from packaging.version import parse


@pytest.mark.parametrize("minor", [10, 14, 15])
@pytest.mark.parametrize("target", ["linux", "darwin", "win32"])
@pytest.mark.parametrize("free_threaded", [False, True])
@pytest.mark.parametrize("rocm", [False, True])
def test_release_wheel_abi(free_threaded, rocm, target, minor):
    source = Path(__file__).resolve().parents[1] / "setup.py"
    tree = ast.parse(source.read_text())
    functions = ast.Module(
        [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"get_platform", "get_wheel_url"}
        ],
        type_ignores=[],
    )
    namespace = {
        "sys": SimpleNamespace(
            platform=target, version_info=SimpleNamespace(major=3, minor=minor)
        ),
        "sysconfig": SimpleNamespace(get_config_var=lambda key: free_threaded),
        "platform": platform,
        "os": __import__("os"),
        "parse": parse,
        "torch": SimpleNamespace(
            __version__="2.14.1",
            version=SimpleNamespace(cuda="13.2"),
            _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True),
        ),
        "IS_ROCM": rocm,
        "HIP_BUILD": rocm,
        "get_hip_version": lambda: parse("7.2"),
        "get_torch_hip_version": lambda: parse("7.2"),
        "get_package_version": lambda: "1.0.0",
        "PACKAGE_NAME": "test_package",
        "BASE_WHEEL_URL": "https://example.org/{tag_name}/{wheel_name}",
    }
    exec(compile(functions, str(source), "exec"), namespace)  # noqa: S102 - controlled setup nodes
    url, filename = namespace["get_wheel_url"]()
    tags = parse_wheel_filename(filename)[3]
    py = f"cp3{minor}"
    assert {tag.interpreter for tag in tags} == {py}
    assert {tag.abi for tag in tags} == {py + ("t" if free_threaded else "")}
    assert url.endswith(filename)
    if target == "win32":
        assert {tag.platform for tag in tags} == {"win_amd64"}
    if target == "darwin":
        assert filename.endswith(f"_{platform.machine()}.whl")


@pytest.mark.parametrize("exit_code,audit_exit", [(0, 0), (9, 0), (0, 7)])
@pytest.mark.parametrize(
    "torch_version", ["2.14.1+cu132", "2.14.0a0+b2c75dd062.nv26.9"]
)
def test_container_build_preserves_torch_and_reports_failure(
    tmp_path, exit_code, audit_exit, torch_version
):
    import json
    import os
    import subprocess

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (tmp_path / "torch.py").write_text(f'__version__ = "{torch_version}"\n')
    commands = {
        "python": f"""#!/bin/bash
if [[ $1 == -m && $2 == pip ]]; then exit 0; fi
if [[ $1 == -m && $2 == build ]]; then
  mkdir -p dist
  printf "%s" "$CAUSAL_CONV1D_LOCAL_VERSION" > build-version
  touch "dist/causal_conv1d-1.0.0+cu13torch2.14cxx11abitrue-cp313-cp313-linux_aarch64.whl"
  exit "$BUILD_EXIT"
fi
if [[ $1 == -m && $2 == auditwheel ]]; then
  printf "%s" "$4" > audit-wheel
  exit "$AUDIT_EXIT"
fi
exec "{sys.executable}" "$@"
""",
        "timeout": '#!/bin/bash\nshift\nexec "$@"\n',
        "uv": f"""#!{sys.executable}
import json, sys
from pathlib import Path
args = sys.argv[1:]
constraint = Path(args[args.index("--excludes") + 1]).read_text()
Path("install.json").write_text(json.dumps([args, constraint]))
""",
    }
    for name, script in commands.items():
        command = bin_dir / name
        command.write_text(script)
        command.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        PYTHONPATH=str(tmp_path),
        BUILD_EXIT=str(exit_code),
        AUDIT_EXIT=str(audit_exit),
        CXX11_ABI="TRUE",
        WHEEL_CUDA_VERSION="13",
        GITHUB_ENV=str(tmp_path / "github-env"),
    )
    env.pop("LD_LIBRARY_PATH", None)
    result = subprocess.run(
        ["bash", str(Path(__file__).resolve().parents[1] / ".github/scripts/build.sh")],
        check=False,
        cwd=tmp_path,
        env=env,
        capture_output=True,
    )
    assert result.returncode == (exit_code or audit_exit), result.stderr.decode()
    args, constraint = json.loads((tmp_path / "install.json").read_text())
    assert "--upgrade" not in args and "--no-sources" in args
    assert "--break-system-packages" in args
    assert args[args.index("--group") + 1] == "build"
    assert constraint == "torch\n"
    assert (tmp_path / "build-version").read_text() == "cu13torch2.14cxx11abiTRUE"
    output = tmp_path / "github-env"
    if exit_code:
        assert not output.exists()
        assert not (tmp_path / "audit-wheel").exists()
    else:
        name = output.read_text().strip().removeprefix("wheel_name=")
        assert parse_wheel_filename(name)[1].local == "cu13torch2.14cxx11abitrue"
        assert (tmp_path / "dist" / name).exists()
        assert (tmp_path / "audit-wheel").read_text() == f"dist/{name}"

    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github/workflows/_build_in_container.yml"
    ).read_text()
    block = re.search(
        r"- name: Build wheel\n.*?        run: \|\n(.*?)(?=\n      -)",
        workflow,
        re.DOTALL,
    ).group(1)
    docker = bin_dir / "docker"
    docker.write_text(
        f'#!/bin/bash\necho "not an exit code"\nexit {exit_code or audit_exit}\n'
    )
    docker.chmod(0o755)
    script = tmp_path / "container.sh"
    script.write_text("set -euo pipefail\n" + textwrap.dedent(block))
    env.update(
        MATRIX_TORCH_VERSION="2.14",
        MATRIX_PYTHON_VERSION="314",
        CI_INPUTS_CONTAINER_IMAGE="test",
    )
    result = subprocess.run(
        ["bash", str(script)], cwd=tmp_path, env=env, check=False, capture_output=True
    )
    assert result.returncode == (exit_code or audit_exit), result.stderr.decode()


@pytest.mark.skipif(not shutil.which("uv"), reason="uv required")
def test_uv_preserves_installed_ngc_torch(tmp_path):
    import zipfile

    version = "2.14.0a0+b2c75dd062.nv26.9"
    with zipfile.ZipFile(
        tmp_path / "ngc_build_tool-1.0-py3-none-any.whl", "w"
    ) as archive:
        info = "ngc_build_tool-1.0.dist-info"
        archive.writestr(
            f"{info}/METADATA",
            "Metadata-Version: 2.1\nName: ngc-build-tool\nVersion: 1.0\n",
        )
        archive.writestr(
            f"{info}/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(
            f"{info}/RECORD", f"{info}/METADATA,,\n{info}/WHEEL,,\n{info}/RECORD,,\n"
        )
    environment = tmp_path / "venv"
    subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(environment)],
        check=True,
        capture_output=True,
    )
    python = environment / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    vendor_torch = next(environment.rglob("site-packages")) / "torch.py"
    vendor_torch.write_text(f'__version__ = "{version}"\n')
    (tmp_path / "pyproject.toml").write_text(
        f'[dependency-groups]\nbuild = ["torch=={version}", "ngc-build-tool"]\n[tool.uv]\nprerelease = "disallow"\n'
    )
    exclude = tmp_path / "torch.txt"
    exclude.write_text("torch\n")
    command = [
        "uv",
        "pip",
        "install",
        "--python",
        str(python),
        "--no-index",
        "--find-links",
        str(tmp_path),
        "--group",
        "build",
    ]
    blocked = subprocess.run(command, cwd=tmp_path, check=False, capture_output=True)
    assert blocked.returncode != 0
    subprocess.run(
        command + ["--excludes", str(exclude)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    assert vendor_torch.read_text() == f'__version__ = "{version}"\n'
    assert not list(environment.rglob("torch-*.dist-info"))


@pytest.mark.parametrize("target", ["linux", "win32"])
def test_host_compiler_flags(target):
    tree = ast.parse((Path(__file__).resolve().parents[1] / "setup.py").read_text())
    assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(item, ast.Name) and item.id == "cxx_flags"
            for item in node.targets
        )
    )
    namespace = {"sys": SimpleNamespace(platform=target)}
    exec(  # noqa: S102 - controlled setup node
        compile(ast.Module([assignment], type_ignores=[]), "setup.py", "exec"),
        namespace,
    )
    flags = namespace["cxx_flags"]
    assert all(flag.startswith("/" if target == "win32" else "-") for flag in flags)
    assert any("c++20" in flag for flag in flags)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "append_nvcc_threads"
    )
    exec(  # noqa: S102 - controlled setup node
        compile(ast.Module([function], type_ignores=[]), "setup.py", "exec"), namespace
    )
    assert ("/Zc:preprocessor" in flags) == (target == "win32")
    assert ("-Xcompiler=/Zc:preprocessor" in namespace["append_nvcc_threads"]([])) == (
        target == "win32"
    )


@pytest.mark.parametrize("failures", [1, 2])
def test_cuda_index_retry_preserves_failure(tmp_path, failures):
    import os

    workflow = (
        Path(__file__).resolve().parents[1] / ".github/workflows/_build.yml"
    ).read_text()
    command = next(
        line.strip()
        for line in workflow.splitlines()
        if "sudo apt-get" in line and "update" in line
    )
    sudo = tmp_path / "sudo"
    sudo.write_text(f"""#!/bin/bash
count=0
if [[ -f "$STATE" ]]; then count=$(cat "$STATE"); fi
count=$((count + 1))
echo "$count" > "$STATE"
if [[ $count -le {failures} ]]; then exit 100; fi
""")
    sudo.chmod(0o755)
    sleep = tmp_path / "sleep"
    sleep.write_text("#!/bin/bash\nexit 0\n")
    sleep.chmod(0o755)
    script = tmp_path / "update.sh"
    script.write_text("set -e\n" + command + "\n")
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        STATE=str(tmp_path / "state"),
    )
    result = subprocess.run(["bash", str(script)], env=env, check=False)
    assert result.returncode == (100 if failures == 2 else 0)
    assert (tmp_path / "state").read_text().strip() == "2"


@pytest.mark.skipif(not shutil.which("pwsh"), reason="PowerShell required")
@pytest.mark.parametrize("version", ["12.8.2", "13.4.2"])
@pytest.mark.parametrize("failure", ["", "signature", "installer"])
def test_windows_cuda_installer(tmp_path, version, failure):
    import os

    workflow = (
        Path(__file__).resolve().parents[1] / ".github/workflows/_build.yml"
    ).read_text()
    block = re.search(
        r"- name: Install CUDA on Windows.*?        run: \|\n(.*?)(?=\n      -)",
        workflow,
        re.DOTALL,
    ).group(1)
    script = tmp_path / "install.ps1"
    script.write_text(
        """
$ErrorActionPreference = 'Stop'
function Invoke-WebRequest { param($Uri, $OutFile)
  $expected = if ($env:CUDA_VERSION -like '13.*') { '_x86_64' } else { '' }
  if ($Uri -notlike "*/cuda_$($env:CUDA_VERSION)_windows$($expected)_network.exe") { throw 'Incorrect installer URL' }
}
function Get-AuthenticodeSignature { param($FilePath)
  @{Status = $(if ($env:TEST_FAILURE -eq 'signature') { 'Invalid' } else { 'Valid' }); SignerCertificate = @{Subject = 'NVIDIA'}}
}
function Start-Process { param($FilePath, $ArgumentList, [switch]$Wait, [switch]$PassThru)
  $minor = ($env:CUDA_VERSION -split '\\.')[0..1] -join '.'
  $packages = "-s nvcc_$minor cudart_$minor thrust_$minor"
  if ($env:CUDA_VERSION -like '13.*') { $packages += " crt_$minor nvvm_$minor" }
  if ($ArgumentList -ne $packages) { throw "Invalid installer components: $ArgumentList" }
  @{ExitCode = $(if ($env:TEST_FAILURE -eq 'installer') { 7 } else { 0 })}
}
function Remove-Item { param($Path) }
"""
        + textwrap.dedent(block)
    )
    env = dict(
        os.environ,
        CUDA_VERSION=version,
        TEST_FAILURE=failure,
        RUNNER_TEMP=str(tmp_path),
        GITHUB_ENV=str(tmp_path / "env"),
        GITHUB_PATH=str(tmp_path / "path"),
    )
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(script)],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    if failure:
        assert result.returncode != 0
        assert (
            "Invalid NVIDIA installer signature"
            if failure == "signature"
            else "CUDA installer failed: 7"
        ) in result.stderr
        assert not (tmp_path / "env").exists()
    else:
        assert result.returncode == 0, result.stderr
        assert f"CUDA\\v{version.rsplit('.', 1)[0]}" in (tmp_path / "env").read_text()
