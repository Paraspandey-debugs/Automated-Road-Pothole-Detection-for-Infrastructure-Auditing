#!/usr/bin/env bash
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

export PYTHONPATH=".:$PYTHONPATH"

echo "================================================================================"
echo "Automated Road Pothole Detection & Infrastructure Severity Auditing"
echo "Master Reproduction Script"
echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Directory: $(pwd)"
echo "Python:    $(python3 --version 2>&1)"
echo "PyTorch:   $(python3 -c 'import torch; print(torch.__version__, "CUDA:" + str(torch.cuda.is_available()))' 2>/dev/null || echo 'Not installed')"
echo "================================================================================"

python3 -u reproduce_all_results.py "$@"

echo ""
echo "All figures generated in: results/figures/"
echo "All tables generated in:  results/tables/"
echo "================================================================================"
