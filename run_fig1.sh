#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/figs"
python3 "fig 1.py" > fig1_output.txt 2>&1
echo "Done at $(date)" >> fig1_output.txt
