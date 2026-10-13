#!/bin/bash

set -eoxu pipefail

python -m pip install --upgrade uv
# NGC's private Torch build is not on PyPI; preserve its ABI.
torch_exclude=$(mktemp)
trap 'rm -f "$torch_exclude"' EXIT
printf 'torch\n' > "$torch_exclude"
uv pip install --system --break-system-packages --group build --no-sources --excludes "$torch_exclude"
export PATH=/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/nvidia/lib64:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

export CAUSAL_CONV1D_FORCE_BUILD="TRUE"
export CAUSAL_CONV1D_FORCE_CXX11_ABI=$CXX11_ABI

torch_version=$(python -c 'import torch; from packaging.version import Version; v = Version(torch.__version__); print(f"{v.major}.{v.minor}")')
export CAUSAL_CONV1D_LOCAL_VERSION="cu${WHEEL_CUDA_VERSION}torch${torch_version}cxx11abi${CXX11_ABI}"

# 5h timeout since GH allows max 6h and we want some buffer
EXIT_CODE=0
timeout 5h python -m build --wheel --no-isolation --outdir dist || EXIT_CODE=$?

if [ "$EXIT_CODE" -eq 0 ]; then
    python - <<'PY'
import os
from pathlib import Path
for wheel in Path("dist").glob("*.whl"):
    package, version, tags = wheel.name.split("-", 2)
    version = version.split("+", 1)[0]
    name = f"{package}-{version}+{os.environ['CAUSAL_CONV1D_LOCAL_VERSION']}-{tags}"
    wheel.rename(wheel.with_name(name))
    with open(os.environ["GITHUB_ENV"], "a") as stream:
        stream.write(f"wheel_name={name}\n")
PY
    for wheel in dist/*.whl; do
        python -m auditwheel show "$wheel"
    done
fi

echo "$EXIT_CODE"
exit "$EXIT_CODE"
