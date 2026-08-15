"""Stage 4 — embed chunks"""
from __future__ import annotations
import time

import numpy as np

from ..contracts import *  # noqa
from ..logging_conf import get_logger

log = get_logger(__name__)

_model_cache: dict[str, object] = {}


def _get_device(embed_cfg: dict) -> str:
    device = embed_cfg.get("device")
    if device:
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _get_embedder(embed_cfg: dict):
    model_name = embed_cfg["model"]
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        t0 = time.time()
        device = _get_device(embed_cfg)
        log.info(f"[embed] loading {model_name} on {device} ...")
        _model_cache[model_name] = SentenceTransformer(model_name, device=device)
        log.info(f"[embed] loaded in {time.time() - t0:.1f}s")
    return _model_cache[model_name]


def embed_texts(texts: list[str], embed_cfg: dict) -> np.ndarray:
    """Low-level helper shared by index/chunk.py's Biography similarity-cut chunker and
    encode() below -- both need the same L2-normalized BGE-M3 embeddings, just at different
    granularities (sentences during chunking vs. final chunks during indexing). This mirrors
    the same shared-embedder structure used in the validated prototype notebook this stage
    was ported from."""
    if not texts:
        return np.zeros((0, embed_cfg.get("dim", 1024)), dtype=np.float32)
    model = _get_embedder(embed_cfg)
    batch_size = embed_cfg.get("batch_size", 32)
    return model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)


def encode(chunks: list[Chunk], cfg: dict):
    """Embed with cfg['embed']['model']. Embeds the citation-header-prefixed Chunk.text
    (already baked in by index/chunk.py's split()) rather than bare dialogue, so act/scene/
    speaker tokens help retrieval too -- see the A2 form, Section 4's Embed row."""
    embed_cfg = cfg["embed"]
    texts = [c.text for c in chunks]
    t0 = time.time()
    log.info(f"[embed] encoding {len(texts)} chunks with {embed_cfg['model']} "
             f"(device={_get_device(embed_cfg)}) ...")
    vectors = embed_texts(texts, embed_cfg)
    log.info(f"[embed] done in {time.time() - t0:.1f}s -> shape {vectors.shape}, dtype {vectors.dtype}")
    assert vectors.shape[0] == len(chunks), "embedding count mismatch"
    return vectors
