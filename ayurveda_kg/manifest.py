"""Dataset manifest: every raw file we depend on is recorded with source, licence, date and checksum."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(manifest_path) -> dict:
    p = Path(manifest_path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def add_entry(manifest_path, name, file_path, source_url, licence) -> dict:
    entry = {
        "file": str(file_path),
        "source_url": source_url,
        "licence": licence,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": sha256_file(file_path),
    }
    m = load(manifest_path)
    m[name] = entry
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")
    return entry


def verify(manifest_path) -> list[str]:
    """Names whose file is missing or whose checksum no longer matches."""
    bad = []
    for name, e in load(manifest_path).items():
        p = Path(e["file"])
        if not p.exists() or sha256_file(p) != e["sha256"]:
            bad.append(name)
    return bad
