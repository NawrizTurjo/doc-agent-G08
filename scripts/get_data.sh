#!/usr/bin/env bash
# A1/A2 — fetch corpus & pre-computed OCR extraction into data/
# Sources:
# 1. Raw Images (300 DPI PNGs): abhishekroy48/complete-works-of-shakespeare -> data/raw/
# 2. Qwen3-VL OCR Transcriptions: abhishekroy48/shakespeare-ocr -> data/ocr_text/
set -euo pipefail

RAW_DATASET="abhishekroy48/complete-works-of-shakespeare"
RAW_DEST="data/raw"

OCR_DATASET="abhishekroy48/shakespeare-ocr"
OCR_DEST="data/ocr_text"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. Install with: pip install kaggle"
  echo "Then place your API token at ~/.kaggle/kaggle.json (kaggle.com -> Account -> Create New Token)."
  exit 1
fi

# 1. Fetch raw page image scans
mkdir -p "$RAW_DEST"
echo "Downloading $RAW_DATASET into $RAW_DEST/..."
kaggle datasets download -d "$RAW_DATASET" -p "$RAW_DEST" --unzip

# 2. Fetch pre-computed Qwen OCR transcripts
mkdir -p "$OCR_DEST"
echo "Downloading $OCR_DATASET into $OCR_DEST/..."
kaggle datasets download -d "$OCR_DATASET" -p "$OCR_DEST" --unzip

echo "Done."
echo "  - Raw Page images are in $RAW_DEST/ (starter_*.png, bio_*.png, page_NNNN.png)."
echo "  - OCR text files are in $OCR_DEST/ (bio/, page/, starter/)."
