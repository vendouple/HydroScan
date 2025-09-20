from __future__ import annotations
import os
import time
import json
import hashlib
from typing import Any


class FileTTLCache:
    def __init__(self, cache_dir: str, ttl_s: int = 86400):
        self.cache_dir = cache_dir
        self.ttl_s = ttl_s
        os.makedirs(self.cache_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"cache_{h}.json")

    def get(self, key: str) -> Any | None:
        p = self._path(key)
        if not os.path.exists(p):
            return None
        try:
            st = os.stat(p)
            if time.time() - st.st_mtime > self.ttl_s:
                return None
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def set(self, key: str, value: Any):
        p = self._path(key)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(value, f)
        os.replace(tmp, p)
