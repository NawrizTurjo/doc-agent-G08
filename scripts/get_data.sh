#!/usr/bin/env bash
# A1 — fetch corpus into data/raw/
# Source: Kaggle dataset of 300 DPI page-image PNGs, re-derived from the P.F. Collier & Son
# Shakespeare scan on Internet Archive (completeworksofw00shakrich). See data/provenance.md.
set -euo pipefail

DATASET="abhishekroy48/complete-works-of-shakespeare"
DEST="data/raw"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. Install with: pip install kaggle"
  echo "Then place your API token at ~/.kaggle/kaggle.json (kaggle.com -> Account -> Create New Token)."
  exit 1
fi

mkdir -p "$DEST"
echo "Downloading $DATASET into $DEST/..."
kaggle datasets download -d "$DATASET" -p "$DEST" --unzip

echo "Done. Page images are in $DEST/ (starter_*.png, bio_*.png, page_NNNN.png)."
