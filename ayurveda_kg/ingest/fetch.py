"""Polite, cached downloader. Raw data is immutable: an existing file is never refetched unless force=True."""
import time
from pathlib import Path

import requests

from ayurveda_kg import manifest

UA = "ayurveda-kg-research/0.1 (academic research; polite rate-limited client)"


def fetch(url, dest, *, name, licence, manifest_path, delay=1.0, session=None, force=False) -> Path:
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    time.sleep(delay)
    r = (session or requests).get(url, headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    if manifest_path:  # bulk crawls pass None and keep their own index
        manifest.add_entry(manifest_path, name, dest, url, licence)
    return dest
