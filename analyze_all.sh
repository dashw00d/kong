#!/usr/bin/env bash
# Analyze all DLLs/EXEs/binaries in a directory with Kong.
# Usage: ./analyze_all.sh /path/to/dlls

set -euo pipefail

DIR="${1:-.}"

if [ ! -d "$DIR" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

export GHIDRA_INSTALL_DIR="${GHIDRA_INSTALL_DIR:-/opt/ghidra_11.3.2_PUBLIC}"
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-21-openjdk-amd64}"

shopt -s nullglob nocaseglob
files=("$DIR"/*.dll "$DIR"/*.exe "$DIR"/*.so "$DIR"/*.dylib "$DIR"/*.bin "$DIR"/*.elf)
shopt -u nullglob nocaseglob

if [ ${#files[@]} -eq 0 ]; then
    echo "No binaries found in $DIR"
    exit 1
fi

echo "Found ${#files[@]} binaries in $DIR"
echo "---"

for f in "${files[@]}"; do
    name=$(basename "$f")
    outdir="$DIR/kong_output_$name"
    echo ""
    echo "=== Analyzing: $name ==="
    uv run kong analyze "$f" --headless -o "$outdir" 2>&1 || echo "FAILED: $name"
    echo "=== Done: $name ==="
done

echo ""
echo "All done. Results in $DIR/kong_output_*/"
