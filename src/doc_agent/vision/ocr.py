"""Stage 3 — OCR/HTR (BASELINE = pretrained foundation, fine-tuned)"""
from __future__ import annotations
import subprocess
from pathlib import Path

from ..contracts import *  # noqa
from ..logging_conf import get_logger
from ..ingest.loader import load_toc, doc_id_for_page

log = get_logger(__name__)


class Reader:
    """Model set by cfg['ocr']. Actual engine: Qwen/Qwen3-VL-4B-Instruct, a vision-language
    model prompted zero-shot to transcribe each page image directly (not fine-tuned) --
    produced this corpus's OCR text (page_0001-page_1318) via offline Kaggle batch runs.
    Traditional OCR engines (Tesseract, PaddleOCR, EasyOCR, docTR) were benchmarked and
    rejected for catastrophic accuracy on this corpus's archaic typography and two-column
    layout -- see notebooks/eda.ipynb Stage 4 for that comparison."""
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["ocr"]

    def transcribe_region(self, region: Region) -> str:
        """Would run live OCR on `region`'s bbox crop of its page image. NOT called by
        transcribe() below: this corpus's OCR already ran once, offline, over the whole
        corpus with reading-order reconstruction already applied (see A1 form Section 2's
        column-reconstruction step) -- re-running a 4B-parameter VLM on ~1,360 pages inside every
        pipeline build would be wasteful and untested against that already-validated
        reconstruction step. Kept as a real method (not deleted) so a live per-region OCR
        path can be dropped in later, e.g. for pages added after this corpus snapshot,
        without changing the Reader interface."""
        raise NotImplementedError(
            "Live per-region OCR isn't wired up on purpose -- this corpus's OCR already ran "
            "offline; transcribe() below reads its output instead of re-running OCR here."
        )


def _ensure_ocr_text_dir(ocr_text_dir: Path, cfg: dict) -> Path:
    """Resolve or auto-download the OCR dataset from Kaggle if not present locally.

    Supports:
    1. Local pre-existing data/ocr_text/ directory.
    2. Kaggle environment mounts (/kaggle/input/shakespeare-ocr or similar).
    3. Auto-download via Kaggle CLI: 'kaggle datasets download -d abhishekroy48/shakespeare-ocr --unzip'
    """
    # 1. Direct local path check (with or without 'shakespeare-ocr' top-level folder)
    if (ocr_text_dir / "page").is_dir() and any((ocr_text_dir / "page").glob("page_*.txt")):
        return ocr_text_dir
    if (ocr_text_dir / "shakespeare-ocr" / "page").is_dir():
        return ocr_text_dir / "shakespeare-ocr"

    # 2. Check Kaggle input mounts if running in a Kaggle notebook environment
    search_roots = [Path("/kaggle/input"), Path(".."), Path("../..")]
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("shakespeare-ocr"):
            if (p / "page").is_dir():
                log.info(f"[ocr] Discovered Kaggle input OCR dataset at: {p}")
                return p
        for p in root.rglob("page"):
            if p.is_dir() and any(p.glob("page_*.txt")):
                log.info(f"[ocr] Discovered Kaggle input OCR text directory at: {p.parent}")
                return p.parent

    # 3. Auto-download via Kaggle CLI
    dataset_name = cfg.get("ocr", {}).get("kaggle_dataset", "abhishekroy48/shakespeare-ocr")
    log.info(f"[ocr] OCR text missing at {ocr_text_dir}. Attempting auto-download of '{dataset_name}'...")
    try:
        ocr_text_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["kaggle", "datasets", "download", "-d", dataset_name, "-p", str(ocr_text_dir), "--unzip"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            log.info(f"[ocr] Successfully downloaded and unzipped '{dataset_name}' into {ocr_text_dir}")
            if (ocr_text_dir / "page").is_dir():
                return ocr_text_dir
            if (ocr_text_dir / "shakespeare-ocr" / "page").is_dir():
                return ocr_text_dir / "shakespeare-ocr"
        else:
            log.warning(f"[ocr] Kaggle CLI returned non-zero ({res.returncode}): {res.stderr.strip()}")
    except Exception as e:
        log.warning(f"[ocr] Automatic Kaggle download failed: {e}. "
                    f"Ensure kaggle CLI is installed ('pip install kaggle') and configured with ~/.kaggle/kaggle.json, "
                    f"or run 'bash scripts/get_data.sh' first.")

    return ocr_text_dir


def _ocr_text_path(page_id: str, ocr_text_dir: Path) -> Path:
    """Locate the .txt file for a given page_id in bio, starter, or page subdirectories."""
    subfolder = "bio" if page_id.startswith("bio_") else "starter" if page_id.startswith("starter_") else "page"

    candidate = ocr_text_dir / subfolder / f"{page_id}.txt"
    if candidate.exists():
        return candidate

    # Fallback if wrapped in a nested 'shakespeare-ocr' directory
    nested_candidate = ocr_text_dir / "shakespeare-ocr" / subfolder / f"{page_id}.txt"
    if nested_candidate.exists():
        return nested_candidate

    # Flat fallback
    flat_candidate = ocr_text_dir / f"{page_id}.txt"
    if flat_candidate.exists():
        return flat_candidate

    return candidate


def transcribe(regions: list[Region], cfg: dict) -> list[Chunk]:
    """Regions -> text chunks.

    This corpus's OCR already ran once, offline (Qwen3-VL-4B-Instruct, prompted zero-shot per
    page). Re-running a 4B-parameter VLM on ~1,360 pages inside every `build_index.sh` call
    would be wasteful and untested against the reading-order-reconstruction step that was
    already validated separately -- so this stage LOADS that
    pre-computed, already-correct-reading-order text per page rather than calling
    Reader.transcribe_region() on pixels (see that method's docstring). Multiple Regions on
    the same page (the two-column split from vision/layout.py) collapse back to ONE Chunk
    per page here, since the pre-computed text file already has both columns concatenated
    in the right order.

    If the OCR text directory is missing, it auto-discovers attached Kaggle inputs or
    auto-downloads the pre-computed dataset 'abhishekroy48/shakespeare-ocr' via the Kaggle API.

    Output is intentionally still PAGE granularity, not final-indexing granularity:
    index/chunk.py re-groups these by work and re-chunks using the hybrid rule-based /
    semantic method, which needs whole-work context (act/scene structure spanning many
    pages) that isn't visible from a single page's Region list.
    """
    ocr_cfg = cfg["ocr"]
    raw_ocr_dir = Path(ocr_cfg.get("ocr_text_dir", "data/ocr_text"))
    ocr_text_dir = _ensure_ocr_text_dir(raw_ocr_dir, cfg)

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
            continue  # skip blank scanner pages
        doc_id = doc_id_for_page(page_id, toc)
        if doc_id is None:
            continue
        chunks.append(Chunk(id=page_id, doc_id=doc_id, text=text, page_ids=[page_id]))

    if missing:
        log.warning(f"[ocr] {missing} page(s) had a Region but no pre-computed OCR text under "
                    f"{ocr_text_dir} -- skipped (run 'bash scripts/get_data.sh' or "
                    f"point ocr_text_dir at its output)")
    log.info(f"[ocr] loaded pre-computed OCR text for {len(chunks)} pages "
             f"({blank} blank pages skipped)")
    return chunks
