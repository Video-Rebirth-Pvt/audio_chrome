# Sourced by every launcher after it cds to the Chrome-10B root.
#   PYTHON=/path/to/python bash scripts/vae/check_h3_parity.sh   # use another interpreter
PYTHON="${PYTHON:-/data/jsfeng/miniconda3/envs/ovi/bin/python}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"
