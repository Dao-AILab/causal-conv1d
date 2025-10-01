#!/bin/bash

set -exou pipefail

pip install dist/*.whl
python -c "import causal_conv1d; print(causal_conv1d.__version__)"