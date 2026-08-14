"""Stage 1 — load scanned page-images"""
from __future__ import annotations
import json
import re
from pathlib import Path

from ..contracts import *  # noqa
from ..logging_conf import get_logger

log = get_logger(__name__)

_BIO_RE = re.compile(r"^bio_(\d+)$")
_PAGE_RE = re.compile(r"^page_(\d+)$")


def load_toc(toc_path: str | Path) -> list[dict]:
    """Shared by ingest/loader.py, vision/layout.py, vision/ocr.py and index/chunk.py --
    toc.json is the single source of truth for which pages belong to which work."""
    with open(toc_path, encoding="utf-8-sig") as f:
        return json.load(f)


def doc_id_for_page(stem: str, toc: list[dict]) -> str | None:
    """Map a page filename stem (e.g. 'page_0500', 'bio_031') to its toc.json work id.
    toc.json has no page_id -> doc_id index; ranges (start_page/end_page for plays/poems,
    the numeric part of start_image/end_image for Biography) are the only lookup key
    available, so every consumer that needs doc_id from a bare filename goes through here."""
    m_bio = _BIO_RE.match(stem)
    if m_bio:
        n = int(m_bio.group(1))
        for work in toc:
            if work.get("category") != "Biography":
                continue
            start = int(re.search(r"\d+", work["start_image"]).group())
            end = int(re.search(r"\d+", work["end_image"]).group())
            if start <= n <= end:
                return str(work["id"])
        return None

    m_page = _PAGE_RE.match(stem)
    if m_page:
        n = int(m_page.group(1))
        for work in toc:
            if "start_page" not in work:
                continue
            if work["start_page"] <= n <= work["end_page"]:
                return str(work["id"])
        return None

    return None  # e.g. starter_*/cover images outside toc.json's numbered sequence -- not routed


def load_pages(cfg: dict) -> list[Page]:
    """Read data/raw/ -> list[Page]. Each page image is matched to its work (doc_id) via
    toc.json's page ranges, since the corpus's own filenames (page_NNNN / bio_NNN) carry no
    work id on their own (see data/provenance.md's split-policy note for the same ranges
    used elsewhere)."""
    corpus_cfg = cfg.get("corpus", {})
    raw_dir = Path(corpus_cfg.get("raw_dir", "data/raw"))
    toc_path = corpus_cfg.get("toc_path", "data/toc.json")
    toc = load_toc(toc_path)

    pages: list[Page] = []
    unrouted = 0
    for image_path in sorted(raw_dir.glob("*.png")):
        stem = image_path.stem
        doc_id = doc_id_for_page(stem, toc)
        if doc_id is None:
            unrouted += 1
            continue
        pages.append(Page(id=stem, image_path=str(image_path), doc_id=doc_id))

    if unrouted:
        log.info(f"[loader] {unrouted} image(s) under {raw_dir} not covered by any toc.json "
                 f"work range (e.g. cover/starter front-matter) -- skipped, not an error")
    log.info(f"[loader] loaded {len(pages)} pages from {raw_dir}")
    return pages
