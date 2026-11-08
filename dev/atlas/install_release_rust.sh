#!/usr/bin/env bash
set -euo pipefail

# Runs inside the manylinux container, with its baseline compiler and linker.
if ! command -v rustup >/dev/null 2>&1; then
    installer=$(mktemp)
    trap 'rm -f "$installer"' EXIT
    curl --proto '=https' --tlsv1.2 --fail --location --retry 3 \
        https://sh.rustup.rs --output "$installer"
    sh "$installer" -y --profile minimal --default-toolchain 1.98.1
fi
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
rustup toolchain install 1.98.1 --profile minimal
cargo +1.98.1 --version
