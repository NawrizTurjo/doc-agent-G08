"""Stage 4 — vector store"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from ..contracts import *  # noqa
from ..logging_conf import get_logger

log = get_logger(__name__)


def _index_dir(cfg: dict) -> Path:
    return Path(cfg["index"].get("index_dir", "data/index"))


def build(chunks: list[Chunk], vectors, cfg: dict) -> None:
    """Persist a vector index (cfg['index']['type']). FAISS IndexFlatIP (exact brute-force
    inner-product search) over L2-normalized embeddings -- A1's committed choice, not the
    starter template's faiss:hnsw placeholder: at ~4,800 vectors x 1024 dims the index is a
    few tens of MB and searches in low-ms on CPU, so there's no recall/latency pressure that
    would justify HNSW's approximate trade-off at this scale (see
    codes/ECI/embed_chunk_index_plan.md S5)."""
    import faiss

    index_type = cfg["index"].get("type", "faiss:flat")
    if index_type != "faiss:flat":
        raise ValueError(f"only 'faiss:flat' is implemented for A2; got {index_type!r}")

    vectors = np.asarray(vectors, dtype=np.float32)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    assert index.ntotal == len(chunks), \
        "FAISS ntotal != chunk count -- the id-mapping (chunk_ids.json) would be misaligned"

    out_dir = _index_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "index.faiss"))

    ids = [c.id for c in chunks]
    with open(out_dir / "chunk_ids.json", "w", encoding="utf-8") as f:
        json.dump(ids, f)

    with open(out_dir / "chunks_contract.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")

    log.info(f"[index] built IndexFlatIP: ntotal={index.ntotal}, dim={dim} -> {out_dir}")


def load(cfg: dict):
    """Load the index + id mapping + chunk objects built by build() above. Returns
    (faiss_index, ids, chunks_by_id) so a caller (e.g. Stage 5 retrieval, or a demo
    notebook) can map a FAISS row back to its Chunk without re-deriving citations."""
    import faiss

    out_dir = _index_dir(cfg)
    index = faiss.read_index(str(out_dir / "index.faiss"))
    with open(out_dir / "chunk_ids.json", encoding="utf-8") as f:
        ids = json.load(f)
    chunks_by_id: dict[str, Chunk] = {}
    with open(out_dir / "chunks_contract.jsonl", encoding="utf-8") as f:
        for line in f:
            c = Chunk.model_validate_json(line)
            chunks_by_id[c.id] = c
    return index, ids, chunks_by_id
