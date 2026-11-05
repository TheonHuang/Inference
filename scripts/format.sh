#!/usr/bin/env bash
# Format all C++/CUDA sources (used by the CI format job).
set -euo pipefail
cd "$(dirname "$0")/.."
find include kernels reference tests bench \
  \( -name "*.h" -o -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" \) \
  -print0 | xargs -0 clang-format -i
echo "formatted."
