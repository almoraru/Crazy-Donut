#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

cc="${CC:-gcc}"
out="${1:-donut}"

if ! command -v "$cc" >/dev/null 2>&1; then
    printf 'Compiler not found: %s\n' "$cc" >&2
    exit 127
fi

"$cc" -O3 -Wall -Wextra -mavx -mfma donut.c -lm -o "$out"
printf 'Built %s\n' "$out"
