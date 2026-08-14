"""Stage 1 — deskew / denoise / binarize / augment"""
from __future__ import annotations
from ..contracts import *  # noqa
from ..logging_conf import get_logger

log = get_logger(__name__)


def run(pages: list[Page], cfg: dict) -> list[Page]:
    """Classical preprocessing. Intentionally a no-op for this corpus: every scan inspected
    (Biography + two-column play pages) showed no visible skew, noise, or fading, and
    single-column pages already reach 2-9% CER with zero cleanup applied (A1 form Section 2,
    'Preparing the scans'). Kept as an explicit pass-through stage -- not removed -- so a
    future scan batch that DOES need deskew/denoise has somewhere to plug in without
    changing the pipeline's shape."""
    log.info(f"[preprocess] no-op pass-through for {len(pages)} pages (clean scans, no cleanup needed)")
    return pages
