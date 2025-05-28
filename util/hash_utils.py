from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(fp: Path | str) -> str:
    """回傳檔案內容的 SHA-256 雜湊值"""
    path = Path(fp)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
