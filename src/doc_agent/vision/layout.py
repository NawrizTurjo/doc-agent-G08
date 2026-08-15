"""Stage 2 — layout detection / segmentation"""
from __future__ import annotations
from ..contracts import *  # noqa
from ..logging_conf import get_logger
from ..ingest.loader import load_toc

log = get_logger(__name__)

# Categories printed in a single column in this edition, per inspection in A1 form Section 2:
# continuous prose (Biography) and the standalone poems/sonnets. Everything else (Comedy/
# History/Tragedy) is the justified two-column dramatic-dialogue layout that the whole A1
# 'data speciality' choice is built around. toc.json carries no per-page layout_type field
# (see the data-schema note in A2_form.md Section 3) so this is a CATEGORY-LEVEL RULE,
# not a real per-page detection -- flagged here for whoever owns Layout/OCR to confirm
# against the actual scans, especially for the Poems (verified for Biography vs. plays in
# A1; not independently re-verified per-page for every poem in this edition).
_SINGLE_COLUMN_CATEGORIES = {"Biography", "Poem"}

_DEFAULT_PAGE_SIZE = (2550, 3300)  # US Letter @ 300 DPI -- this edition's rendering default (A1 Section 5)


def _page_size(image_path: str) -> tuple[int, int]:
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            return im.size  # (width, height)
    except Exception:
        log.warning(f"[layout] could not read image size for {image_path!r}, using default page size")
        return _DEFAULT_PAGE_SIZE


def detect(pages: list[Page], cfg: dict) -> list[Region]:
    """Detect text/table/figure/heading regions.

    For this corpus the real layout problem isn't finding text vs. non-text (every page is
    >=95% text, no illustrations per A1 Section 2) -- it's recovering correct READING ORDER
    on two-column pages, which A1's own measurement showed is the single biggest lever on
    transcription quality (CER ~0.77 -> ~0.04 once columns are split at the gutter, A1 form
    Section 2/3). So this emits ONE region for a single-column page, or TWO regions (left of
    the gutter, right of the gutter) for a two-column page -- kind='text' for both; table/
    figure kinds aren't needed on this corpus.
    """
    toc_path = cfg.get("corpus", {}).get("toc_path", "data/toc.json")
    toc = load_toc(toc_path)
    works_by_id = {str(w["id"]): w for w in toc}

    regions: list[Region] = []
    n_two_col = 0
    for page in pages:
        work = works_by_id.get(page.doc_id)
        category = work["category"] if work else None
        is_single_column = category in _SINGLE_COLUMN_CATEGORIES

        width, height = _page_size(page.image_path)
        if is_single_column:
            regions.append(Region(page_id=page.id, bbox=(0, 0, width, height), kind="text"))
        else:
            n_two_col += 1
            gutter_x = width // 2
            regions.append(Region(page_id=page.id, bbox=(0, 0, gutter_x, height), kind="text"))
            regions.append(Region(page_id=page.id, bbox=(gutter_x, 0, width, height), kind="text"))

    log.info(f"[layout] {len(pages)} pages -> {len(regions)} regions "
             f"({n_two_col} two-column pages, {len(pages) - n_two_col} single-column)")
    return regions
