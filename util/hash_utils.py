from __future__ import annotations

import hashlib

def sha256_file(data: bytes) -> str:
    """回傳影像資料的 SHA-256 雜湊值"""
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()
