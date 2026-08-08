"""Data — data schema/quality validation at ingest"""
from __future__ import annotations
from ..contracts import *  # noqa

MIN_PAGES = 300  # course floor (data/README.md)
# The 60,000-word floor is NOT checked here: Page carries no text field (only id,
# image_path, doc_id) — word count only exists on Chunk.text after OCR (Stage 3/4).
# Documented pre-OCR word estimate lives in data/provenance.md and notebooks/eda.ipynb.
# Split-leakage is likewise not checkable from a bare list[Page] (no split label in the
# contract); the whole-document split + its no-leakage assertion is built in
# notebooks/eda.ipynb against data/toc.json, per data/provenance.md's split policy.

def validate(pages: list[Page]) -> None:
    """Assert min page count and per-page format/uniqueness. IMPLEMENT."""
    if len(pages) < MIN_PAGES:
        raise ValueError(f"corpus has {len(pages)} pages, need >= {MIN_PAGES}")

    seen_ids: set[str] = set()
    for p in pages:
        if not p.id or not p.doc_id or not p.image_path:
            raise ValueError(f"Page {p.id!r} missing id/doc_id/image_path")
        if p.id in seen_ids:
            raise ValueError(f"duplicate page id: {p.id}")
        seen_ids.add(p.id)

