"""Stage 3 — OCR/HTR (BASELINE = pretrained foundation, fine-tuned)"""
from __future__ import annotations
from pathlib import Path

from ..contracts import *  # noqa
from ..logging_conf import get_logger
from ..ingest.loader import load_toc, doc_id_for_page

log = get_logger(__name__)


class Reader:
    """Model set by cfg['ocr']. Actual engine: Qwen/Qwen3-VL-4B-Instruct, a vision-language
    model prompted zero-shot to transcribe each page image directly (not fine-tuned) -- see
    codes/q3_4B_gt_part1.ipynb through part4.ipynb for the 4-notebook Kaggle batch run that
    produced this corpus's OCR text (page_0001-page_1318). Traditional OCR engines
    (Tesseract, PaddleOCR, EasyOCR, docTR) were benchmarked and rejected for catastrophic
    accuracy on this corpus's archaic typography and two-column layout -- see
    notebooks/eda.ipynb Stage 4 for that comparison."""
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["ocr"]

    def transcribe_region(self, region: Region) -> str:
        """Would run live OCR on `region`'s bbox crop of its page image. NOT called by
        transcribe() below: this corpus's OCR already ran once, offline, over the whole
        corpus with reading-order reconstruction already applied (see A1 form Section 2's
        column-reconstruction step and codes/q3_4B_gt_part1..4.ipynb for the actual
        production OCR run) -- re-running a 4B-parameter VLM on ~1,360 pages inside every
        pipeline build would be wasteful and untested against that already-validated
        reconstruction step. Kept as a real method (not deleted) so a live per-region OCR
        path can be dropped in later, e.g. for pages added after this corpus snapshot,
        without changing the Reader interface."""
        raise NotImplementedError(
            "Live per-region OCR isn't wired up on purpose -- this corpus's OCR already ran "
            "offline; transcribe() below reads its output instead of re-running OCR here."
        )


def _ocr_text_path(page_id: str, ocr_text_dir: Path) -> Path:
    if page_id.startswith("bio_"):
        return ocr_text_dir / "bio" / f"{page_id}.txt"
    return ocr_text_dir / "page" / f"{page_id}.txt"


def transcribe(regions: list[Region], cfg: dict) -> list[Chunk]:
    """Regions -> text chunks.

    This corpus's OCR already ran once, offline (Qwen3-VL-4B-Instruct, prompted zero-shot per
    page across codes/q3_4B_gt_part1..4.ipynb). Re-running a 4B-parameter VLM on ~1,360 pages inside every `build_index.sh` call
    would be wasteful and untested against the reading-order-reconstruction step that was
    already validated separately -- so this stage LOADS that
    pre-computed, already-correct-reading-order text per page rather than calling
    Reader.transcribe_region() on pixels (see that method's docstring). Multiple Regions on
    the same page (the two-column split from vision/layout.py) collapse back to ONE Chunk
    per page here, since the pre-computed text file already has both columns concatenated
    in the right order.

    Output is intentionally still PAGE granularity, not final-indexing granularity:
    index/chunk.py re-groups these by work and re-chunks using the hybrid rule-based /
    semantic method, which needs whole-work context (act/scene structure spanning many
    pages) that isn't visible from a single page's Region list.
    """
    ocr_cfg = cfg["ocr"]
    ocr_text_dir = Path(ocr_cfg.get("ocr_text_dir", "data/ocr_text"))
    toc_path = cfg.get("corpus", {}).get("toc_path", "data/toc.json")
    toc = load_toc(toc_path)

    page_ids = sorted({region.page_id for region in regions})

    chunks: list[Chunk] = []
    missing = 0
    blank = 0
    for page_id in page_ids:
        text_path = _ocr_text_path(page_id, ocr_text_dir)
        if not text_path.exists():
            missing += 1
            continue
        text = text_path.read_text(encoding="utf-8", errors="ignore")
        if text.strip() == "[Blank Page]":
            blank += 1
            continue  # matches codes/ECI/chunk_reference_impl.py's blank-page skip
        doc_id = doc_id_for_page(page_id, toc)
        if doc_id is None:
            continue
        chunks.append(Chunk(id=page_id, doc_id=doc_id, text=text, page_ids=[page_id]))

    if missing:
        log.warning(f"[ocr] {missing} page(s) had a Region but no pre-computed OCR text under "
                    f"{ocr_text_dir} -- skipped (run codes/final_texts OCR pipeline first, or "
                    f"point ocr_text_dir at its output)")
    log.info(f"[ocr] loaded pre-computed OCR text for {len(chunks)} pages "
             f"({blank} blank pages skipped)")
    return chunks
