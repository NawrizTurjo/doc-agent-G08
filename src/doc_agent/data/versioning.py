"""Data — corpus versioning (which corpus version -> which result)"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from ..contracts import *  # noqa

VERSION_LOG = Path("data/versions.jsonl")  # append-only: which corpus version -> which result

def snapshot(corpus_dir: str) -> str:
    """Hash + record a corpus version id. IMPLEMENT (or wire DVC)."""
    root = Path(corpus_dir)
    files = sorted(p for p in root.rglob("*") if p.is_file())

    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(root).as_posix().encode())
        h.update(str(f.stat().st_size).encode())
    version_id = h.hexdigest()[:12]

    VERSION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with VERSION_LOG.open("a") as fh:
        fh.write(json.dumps({"corpus_dir": str(root), "version_id": version_id, "n_files": len(files)}) + "\n")
    return version_id

