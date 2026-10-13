#!/bin/bash

set -exou pipefail

pip install dist/*.whl
(cd /tmp && python -c "import causal_conv1d, causal_conv1d_cuda; print(causal_conv1d.__version__)")