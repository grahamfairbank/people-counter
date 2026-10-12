"""Runtime settings: defaults come from the environment, UI edits persist to /data."""
import json
import os
import re
import threading
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))

# name -> (env var, type, default, min, max)
FIELDS = {
    "source": ("RTSP_URL", str, "", None, None),
    "fps": ("TARGET_FPS", float, 5.0, 1, 15),
    "conf": ("CONFIDENCE", float, 0.45, 0.1, 0.95),
    "min_hits": ("MIN_HITS", int, 5, 1, 100),
    "reentry_minutes": ("REENTRY_MINUTES", float, 5.0, 0, 240),
    "similarity": ("SIMILARITY", float, 0.7, 0.1, 1.0),
    "quick_seconds": ("QUICK_RETURN_SECONDS", float, 30.0, 0, 600),
}


def mask_url(url: str) -> str:
    """Hide the password of rtsp://user:pass@host/... for display."""
    return re.sub(r"(://[^:/@]+:)[^@]+@", r"\1***@", url)


class Config:
    def __init__(self):
        self._lock = threading.Lock()
        self._path = DATA_DIR / "config.json"
        self.version = 0
        for name, (env, typ, default, _, _) in FIELDS.items():
            setattr(self, name, typ(os.environ.get(env, default)))
        if self._path.exists():
            self._apply(json.loads(self._path.read_text()))

    def _apply(self, values: dict) -> bool:
        changed = False
        for name, value in values.items():
            if name not in FIELDS or value is None:
                continue
            _, typ, _, lo, hi = FIELDS[name]
            value = typ(value)
            if typ is str:
                value = value.strip()
            else:
                value = max(lo, min(hi, value))
            if getattr(self, name) != value:
                setattr(self, name, value)
                changed = True
        return changed

    def update(self, values: dict) -> bool:
        with self._lock:
            # The UI only ever sees the masked URL; sending it back means "unchanged".
            if values.get("source") == mask_url(self.source):
                values = {k: v for k, v in values.items() if k != "source"}
            changed = self._apply(values)
            if changed:
                self.version += 1
                self._path.write_text(json.dumps(self.as_dict(masked=False), indent=2))
            return changed

    def as_dict(self, masked: bool = True) -> dict:
        out = {name: getattr(self, name) for name in FIELDS}
        if masked:
            out["source"] = mask_url(out["source"])
        return out
