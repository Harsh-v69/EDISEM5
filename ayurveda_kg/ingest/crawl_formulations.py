"""Phase 5 crawl: every IMPPAT formulation page + therapeutic-use pages of the scoped herbs. Resumable, 1 request/second, cached."""
import csv
import hashlib
import urllib.parse
from pathlib import Path

from ayurveda_kg.ingest.fetch import fetch
from ayurveda_kg.ingest.imppat import BASE, LICENCE, formulation_links
from ayurveda_kg.scope import load_scope

RAW = Path("data/raw/imppat")


def safe_text(msg: str) -> str:
    """Console-safe log text: Sanskrit names must never crash the error handler (Windows consoles are cp1252)."""
    return msg.encode("ascii", "backslashreplace").decode("ascii")


def main(delay=1.0):
    home = fetch(BASE, RAW / "home.html", name="imppat_home", licence=LICENCE, manifest_path=None, delay=delay)
    links = formulation_links(home.read_text(encoding="utf-8", errors="ignore"))
    print(f"{len(links)} formulation pages", flush=True)
    for h in load_scope()["herbs"]:
        fetch(f"{BASE}therapeutics/{urllib.parse.quote(h['imppat_name'])}", RAW / "therapeutics" / f"{h['imppat_name'].replace(' ', '_')}.html",
              name=f"therapeutics_{h['imppat_name']}", licence=LICENCE, manifest_path=None, delay=delay)
    ok = failed = 0
    index = []
    for i, (kind, name) in enumerate(links, 1):
        fn = f"{kind}_{hashlib.md5(name.encode('utf-8')).hexdigest()[:12]}.html"
        try:
            fetch(f"{BASE}{kind}formulationdetails/{urllib.parse.quote(name)}", RAW / "formulations" / fn,
                  name=name, licence=LICENCE, manifest_path=None, delay=delay)
            index.append((kind, name, fn)); ok += 1
        except Exception as e:
            failed += 1
            print(safe_text(f"FAILED {kind} {name}: {e!r}"), flush=True)
        if i % 100 == 0:
            print(f"[formulations] {i}/{len(links)} ok={ok} failed={failed}", flush=True)
    with open(RAW / "formulations_index.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["kind", "name", "file"]); w.writerows(index)
    print({"ok": ok, "failed": failed}, flush=True)


if __name__ == "__main__":
    main()
