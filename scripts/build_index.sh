#!/usr/bin/env bash
# A2 — build the vector index end to end.
set -euo pipefail

# The starter's original version of this script ran `make ingest index`. That runs BOTH the
# `ingest` and `index` Makefile targets, i.e. `scripts/run_ingest.py` AND `scripts/run_index.py`
# -- but both of those call the exact same `pipeline.build_knowledge_base(cfg)` (indexing is
# already embedded inside that one call; run_index.py's own comment says as much: "index is
# built inside build_knowledge_base; kept for staged runs"). So the original one-liner ran the
# FULL heavy pipeline (OCR load, chunk, embed, index) twice in a row for no benefit. Calling
# `run_ingest.py` once is equivalent and roughly halves build time.
python scripts/run_ingest.py
