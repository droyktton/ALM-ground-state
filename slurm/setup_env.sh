#!/bin/bash
# One-off, run-from-the-frontend helper: make sure the "alm_env" conda env
# has everything the project needs (numpy/scipy/matplotlib always; pandas is
# only needed for plot_zeta_vs_n.py). Safe to re-run.
set -euo pipefail

source /home/koltona/miniconda3/etc/profile.d/conda.sh
conda activate alm_env

python3 -m pip install --upgrade pip >/dev/null
python3 -m pip install numpy scipy matplotlib pandas

echo "alm_env packages:"
python3 -c "
import numpy, scipy, matplotlib, pandas
print('numpy      ', numpy.__version__)
print('scipy      ', scipy.__version__)
print('matplotlib ', matplotlib.__version__)
print('pandas     ', pandas.__version__)
"
