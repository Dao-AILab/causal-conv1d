# Copyright (c) 2024, Tri Dao.

import sys
import warnings
import os
import re
import shutil
import ast
from pathlib import Path
from packaging.version import parse, Version
import platform
import subprocess
import urllib.request
import urllib.error
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

from setuptools import setup, find_packages

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME, HIP_HOME

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "causal_conv1d"

BASE_WHEEL_URL = "https://github.com/Dao-AILab/causal-conv1d/releases/download/{tag_name}/{wheel_name}"

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("CAUSAL_CONV1D_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("CAUSAL_CONV1D_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("CAUSAL_CONV1D_FORCE_CXX11_ABI", "FALSE") == "TRUE"


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return "linux_x86_64"
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


def get_cuda_bare_metal_version(cuda_dir):
    # Build nvcc executable path in an OS-independent way.
    nvcc_executable = os.path.join(cuda_dir, "bin", "nvcc")
    if sys.platform == "win32":
        nvcc_executable += ".exe"
    print(f"nvcc_executable = {nvcc_executable}")
    raw_output = subprocess.check_output([nvcc_executable, "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])
    return raw_output, bare_metal_version


def get_hip_version(rocm_dir):
    hipcc_bin = "hipcc" if rocm_dir is None else os.path.join(rocm_dir, "bin", "hipcc")
    try:
        raw_output = subprocess.check_output([hipcc_bin, "--version"], universal_newlines=True)
    except Exception as e:
        print(
            f"hip installation not found: {e} ROCM_PATH={os.environ.get('ROCM_PATH')}"
        )
        return None, None

    for line in raw_output.split("\n"):
        if "HIP version" in line:
            rocm_version = parse(line.split()[-1].replace("-", "+"))
            return line, rocm_version

    return None, None


def get_torch_hip_version():
    if torch.version.hip:
        return parse(torch.version.hip.split()[-1].replace("-", "+"))
    else:
        return None


def check_if_hip_home_none(global_option: str) -> None:
    if HIP_HOME is not None:
        return
    warnings.warn(
        f"{global_option} was requested, but hipcc was not found.  Are you sure your environment has hipcc available?"
    )


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", "4"]


cmdclass = {}
ext_modules = []

HIP_BUILD = bool(torch.version.hip)

if not SKIP_CUDA_BUILD:

    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    cc_flag = []

    if HIP_BUILD:
        check_if_hip_home_none(PACKAGE_NAME)
        rocm_home = os.getenv("ROCM_PATH")
        _, hip_version = get_hip_version(rocm_home)

        if HIP_HOME is not None:
            if hip_version < Version("6.0"):
                raise RuntimeError(
                    f"{PACKAGE_NAME} is only supported on ROCm 6.0 and above.  "
                    "Note: make sure HIP has a supported version by running hipcc --version."
                )
            if hip_version == Version("6.0"):
                warnings.warn(
                    f"{PACKAGE_NAME} requires a patch to be applied when running on ROCm 6.0. "
                    "Refer to the README.md for detailed instructions.",
                    UserWarning
                )

        cc_flag.append("-DBUILD_PYTHON_PACKAGE")
    else:
        check_if_cuda_home_none(PACKAGE_NAME)
        if CUDA_HOME is not None:
            _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
            if bare_metal_version < Version("11.6"):
                raise RuntimeError(
                    f"{PACKAGE_NAME} is only supported on CUDA 11.6 and above.  "
                    "Note: make sure nvcc has a supported version by running nvcc -V."
                )
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_53,code=sm_53")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_62,code=sm_62")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_70,code=sm_70")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_72,code=sm_72")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_87,code=sm_87")
        if CUDA_HOME is not None and bare_metal_version >= Version("11.8"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_90,code=sm_90")
        if bare_metal_version >= Version("12.8"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_100,code=sm_100")

    # Set the C++ optimization flag appropriately for the platform.
    if sys.platform == "win32":
        cxx_opt = ["/O2", "/Zc:__cplusplus", "/std:c++17", "/FIiso646.h"]
    else:
        cxx_opt = ["-O3"]

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True

    if HIP_BUILD:
        extra_compile_args = {
            "cxx": cxx_opt + ["-std=c++17"],
            "nvcc": [
                        "-O3",
                        "-std=c++17",
                        f"--offload-arch={os.getenv('HIP_ARCHITECTURES', 'native')}",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-fgpu-flush-denormals-to-zero",
                        "-allow-unsupported-compiler"
                    ] + cc_flag,
        }
    else:
        extra_compile_args = {
            "cxx": cxx_opt,
            "nvcc": append_nvcc_threads(
                [
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                    "-lineinfo",
                    "-allow-unsupported-compiler"
                ] + cc_flag
            ),
        }

    ext_modules.append(
        CUDAExtension(
            name="causal_conv1d_cuda",
            sources=[
                "csrc/causal_conv1d.cpp",
                "csrc/causal_conv1d_fwd.cu",
                "csrc/causal_conv1d_bwd.cu",
                "csrc/causal_conv1d_update.cu",                
            ],
            extra_compile_args=extra_compile_args,
            include_dirs=["csrc/causal_conv1d"],
        )
    )


def get_package_version():
    with open(Path(this_dir) / "causal_conv1d" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("CAUSAL_CONV1D_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)


def get_wheel_url():
    # Determine the version numbers that will be used to determine the correct wheel
    torch_version_raw = parse(torch.__version__)

    if HIP_BUILD:
        torch_hip_version = get_torch_hip_version()
        hip_version = f"{torch_hip_version.major}{torch_hip_version.minor}"
    else:
        torch_cuda_version = parse(torch.version.cuda)
        torch_cuda_version = parse("11.8") if torch_cuda_version.major == 11 else parse("12.3")
        cuda_version = f"{torch_cuda_version.major}"

    gpu_compute_version = hip_version if HIP_BUILD else cuda_version
    cuda_or_hip = "hip" if HIP_BUILD else "cu"

    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_name = get_platform()
    causal_conv1d_version = get_package_version()

    torch_version = f"{torch_version_raw.major}.{torch_version_raw.minor}"
    cxx11_abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()

    wheel_filename = f"{PACKAGE_NAME}-{causal_conv1d_version}+{cuda_or_hip}{gpu_compute_version}torch{torch_version}cxx11abi{cxx11_abi}-{python_version}-{python_version}-{platform_name}.whl"

    wheel_url = BASE_WHEEL_URL.format(
        tag_name=f"v{causal_conv1d_version}", wheel_name=wheel_filename
    )
    return wheel_url, wheel_filename


class CachedWheelsCommand(_bdist_wheel):
    """
    The CachedWheelsCommand plugs into the default bdist wheel, which is ran by pip when it cannot
    find an existing wheel (which is currently the case for all installs). We use
    the environment parameters to detect whether there is already a pre-built version of a compatible
    wheel available and short-circuits the standard full build pipeline.
    """
    def run(self):
        if FORCE_BUILD:
            return super().run()

        wheel_url, wheel_filename = get_wheel_url()
        print("Guessing wheel URL: ", wheel_url)
        try:
            urllib.request.urlretrieve(wheel_url, wheel_filename)

            # Make the archive
            # Lifted from the root wheel processing command
            # https://github.com/pypa/wheel/blob/cf71108ff9f6ffc36978069acb28824b44ae028e/src/wheel/bdist_wheel.py#LL381C9-L381C85
            if not os.path.exists(self.dist_dir):
                os.makedirs(self.dist_dir)

            impl_tag, abi_tag, plat_tag = self.get_tag()
            archive_basename = f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"

            wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
            print("Raw wheel path", wheel_path)
            shutil.move(wheel_filename, wheel_path)
        except urllib.error.HTTPError:
            print("Precompiled wheel not found. Building from source...")
            # If the wheel could not be downloaded, build from source
            super().run()


setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
            "causal_conv1d.egg-info",
        )
    ),
    author="Tri Dao",
    author_email="tri@tridao.me",
    description="Causal depthwise conv1d in CUDA, with a PyTorch interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dao-AILab/causal-conv1d",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": CachedWheelsCommand, "build_ext": BuildExtension} if ext_modules else {"bdist_wheel": CachedWheelsCommand},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.4.0",
        "einops",
        "packaging",
        "ninja"
    ]
)
