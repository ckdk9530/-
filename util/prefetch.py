from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, Iterable, Tuple, Optional
import hashlib
import io

from PIL import Image


class ImagePrefetcher:
    """簡單的影像預讀快取，預設保留 5 張。"""

    def __init__(self, size: int = 5) -> None:
        self.size = size
        self._cache: Dict[Path, Tuple[bytes, str]] = {}
        self._order: deque[Path] = deque()

    def prefetch(self, paths: Iterable[Path | str]) -> None:
        """將給定路徑的圖片讀入快取，超出大小時會自動淘汰最舊項目"""
        for p in paths:
            path = Path(p)
            if path in self._cache:
                continue
            try:
                data = path.read_bytes()
                sha = hashlib.sha256(data).hexdigest()
            except Exception:
                continue
            if len(self._order) >= self.size:
                old = self._order.popleft()
                self._cache.pop(old, None)
            self._cache[path] = (data, sha)
            self._order.append(path)

    def pop_image(self, p: Path | str) -> Optional[Image.Image]:
        """取得並移除快取中的圖片，如不存在則回傳 None"""
        path = Path(p)
        entry = self._cache.pop(path, None)
        if entry is None:
            return None
        self._order.remove(path)
        data, _sha = entry
        return Image.open(io.BytesIO(data)).convert("RGB")

    def get_sha(self, p: Path | str) -> Optional[str]:
        path = Path(p)
        entry = self._cache.get(path)
        return entry[1] if entry else None
