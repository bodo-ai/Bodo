#!/bin/bash
set -euo pipefail

# Target width and height
WIDTH=1920
HEIGHT=1080

# Output directory (keeps originals untouched)
mkdir -p resized

for f in *.png; do
    out="resized/$f"
    echo "Resizing $f -> $out"
    convert "$f" -resize "${WIDTH}x${HEIGHT}!" "$out"
done

echo "Done. Resized images are in ./resized/"

