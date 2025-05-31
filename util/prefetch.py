from __future__ import annotations

from collections import deque
import threading
from pathlib import Path
from typing import Dict, Iterable, Optional
import io

from PIL import Image


class ImagePrefetcher:
    """簡單的影像預讀快取，預設保留 5 張。"""

    def __init__(self, size: int = 5, workers: int | None = None) -> None:
        self.size = size
        self._cache: Dict[Path, bytes] = {}
        self._order: deque[Path] = deque()
        self._lock = threading.Lock()

    def prefetch(self, paths: Iterable[Path | str]) -> None:
        """將給定路徑的圖片讀入快取，超出大小時會自動淘汰最舊項目"""

        targets: list[Path] = []
        for p in paths:
            path = Path(p)
            if path in self._cache or path in targets:
                continue
            targets.append(path)

        def _load(path: Path):
            try:
                data = path.read_bytes()
                return path, data
            except Exception:
                return None

        for result in map(_load, targets):
            if result is None:
                continue
            path, data = result
            with self._lock:
                if path in self._cache:
                    continue
                if len(self._order) >= self.size:
                    old = self._order.popleft()
                    self._cache.pop(old, None)
                self._cache[path] = data
                self._order.append(path)

    def pop_image(self, p: Path | str) -> Optional[Image.Image]:
        """取得並移除快取中的圖片，如不存在則回傳 None"""
        path = Path(p)
        with self._lock:
            data = self._cache.pop(path, None)
            if data is None:
                return None
            self._order.remove(path)
        return Image.open(io.BytesIO(data)).convert("RGB")
