"""IMPPAT page parsers (pure functions) and a polite crawler. IMPPAT is CC BY-NC-ND: never redistribute its tables."""
import re

from bs4 import BeautifulSoup


def _soup(html):
    return BeautifulSoup(html, "lxml")


def _cells(tr):
    return [c.get_text(" ", strip=True) for c in tr.find_all("td")]


def parse_plant_page(html) -> list[dict]:
    table = _soup(html).find("table", class_="phytochem")
    if table is None:
        return []
    rows = []
    for tr in table.find_all("tr"):
        c = _cells(tr)
        if len(c) >= 5 and c[2].startswith("IMPPAT"):
            rows.append({"plant": c[0], "part": c[1], "phy_id": c[2], "phy_name": c[3], "reference": c[4]})
    return rows


def _after(text, label, stops):
    m = re.search(re.escape(label) + r"\s*(.*?)\s*(?:" + "|".join(map(re.escape, stops)) + r"|$)", text, re.S)
    return m.group(1).strip() if m else ""


def parse_detail_page(html) -> dict:
    text = re.sub(r"\s+", " ", _soup(html).get_text(" ", strip=True))
    ext = _after(text, "External chemical identifiers:", ["Chemical structure information"])
    syn = _after(text, "Synonymous chemical names:", ["External chemical identifiers:"])
    def ext_id(prefix):
        m = re.search(prefix + r"[:_]\s*(?:CID_)?([A-Za-z0-9_]+)", ext)
        return m.group(1) if m else ""
    cid = re.search(r"CID:CID_(\d+)", ext)
    return {
        "phy_id": _after(text, "IMPPAT Phytochemical identifier:", ["Phytochemical name:"]),
        "name": _after(text, "Phytochemical name:", ["Synonymous chemical names:", "External chemical identifiers:"]),
        "synonyms": [s.strip() for s in syn.split(",") if s.strip()],
        "cid": cid.group(1) if cid else "",
        "chembl_id": ext_id("ChEMBL"),
        "smiles": _after(text, "SMILES:", ["InChI:"]),
        "inchi": _after(text, "InChI:", ["InChIKey:"]),
        "inchikey": _after(text, "InChIKey:", ["DeepSMILES:"]),
    }


def parse_targets_page(html) -> dict:
    targets, adme = [], {}
    for tr in _soup(html).find_all("tr"):
        c = _cells(tr)
        if len(c) == 5 and c[0].startswith("TAR_"):
            targets.append({"gene": c[2], "ensembl": c[1], "entrez": c[3], "source": c[4]})
        elif len(c) == 3 and re.search(r"(inhibitor|substrate)$", c[0]):
            adme[c[0]] = c[2]
    return {"targets": targets, "adme": adme}


# ---- crawler ----
import csv
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from ayurveda_kg import manifest as _manifest
from ayurveda_kg.ingest.fetch import fetch

BASE = "https://cb.imsc.res.in/imppat/"
LICENCE = "CC BY-NC-ND 4.0 (IMSc); do not redistribute"


def plant_ids(rows) -> list[str]:
    return sorted({r["phy_id"] for r in rows})


def crawl_plants(herb_names, raw_dir, delay=1.0, session=None) -> dict:
    out = {}
    for h in herb_names:
        out[h] = fetch(BASE + "phytochemical/" + urllib.parse.quote(h), Path(raw_dir) / "plants" / f"{h.replace(' ', '_')}.html",
                       name=h, licence=LICENCE, manifest_path=None, delay=delay, session=session)
    return out


def crawl_compounds(phy_ids, raw_dir, delay=1.0, session=None, manifest_path="data/manifest.json", log_every=100) -> dict:
    """Download detail + humantargets page per compound. Failures are logged, not fatal (hour-long run); resumable."""
    raw = Path(raw_dir)
    ok = failed = 0
    failures = []
    for i, pid in enumerate(phy_ids, 1):
        for kind, path in (("phytochemical-detailedpage", "detail"), ("humantargets", "targets")):
            dest = raw / "compounds" / f"{pid}.{path}.html"
            try:
                fetch(f"{BASE}{kind}/{pid}", dest, name=f"{pid}.{path}", licence=LICENCE, manifest_path=None, delay=delay, session=session)
                ok += 1
            except Exception as e:
                failed += 1
                failures.append((pid, path, repr(e)))
        if log_every and i % log_every == 0:
            print(f"[crawl] {i}/{len(phy_ids)} compounds, ok={ok} failed={failed}", flush=True)
    idx_path = raw / "crawl_index.tsv"
    with open(idx_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["path", "sha256", "bytes"])
        for p in sorted((raw / "compounds").glob("*.html")):
            w.writerow([p.name, _manifest.sha256_file(p), p.stat().st_size])
    if failures:
        (raw / "crawl_failures.tsv").write_text("\n".join("\t".join(x) for x in failures), encoding="utf-8")
    if manifest_path:
        _manifest.add_entry(manifest_path, "imppat_crawl_index", idx_path, BASE, LICENCE)
    return {"ok": ok, "failed": failed}


# ---- formulations and therapeutic uses (Phase 5) ----
def parse_formulation_page(html) -> dict:
    """Ingredient list of one IMPPAT formulation. IMPPAT gives NO proportions, only ingredient name and plant part."""
    soup = _soup(html)
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    ing = []
    for table in soup.find_all("table"):
        head = [th.get_text(" ", strip=True) for th in table.find_all("th")]
        if head[:3] == ["Formulation name", "Ingredient name", "Plant part"]:
            for tr in table.find_all("tr"):
                c = _cells(tr)
                if len(c) >= 2 and c[1]:
                    ing.append({"ingredient": c[1], "part": c[2] if len(c) > 2 else ""})
            break
    ident = re.search(r"Formulation identifier:\s*([A-Z]+\d+)", text)
    dose = re.search(r"Dosage \(according to ([^)]*)\):\s*(.*?)(?:\s+Ingredients of|$)", text)
    name = ""
    h = re.search(r"Ingredients of (.*?) Formulation name", text)
    if h:
        name = h.group(1).strip()
    return {"id": ident.group(1) if ident else "", "name": name, "ingredients": ing,
            "dosage": dose.group(2).strip() if dose else ""}


def parse_therapeutics_page(html) -> list[str]:
    """Therapeutic uses reported for one plant (IMPPAT 'therapeutics' page)."""
    uses = []
    for table in _soup(html).find_all("table"):
        head = [th.get_text(" ", strip=True) for th in table.find_all("th")]
        if any("Therapeutic use" in h for h in head):
            col = next(i for i, h in enumerate(head) if "Therapeutic use" in h)
            for tr in table.find_all("tr"):
                c = _cells(tr)
                if len(c) > col and c[col]:
                    uses.append(c[col])
    return sorted(set(uses))


def formulation_names(home_html) -> list[str]:
    """All formulation page names linked from the IMPPAT home page (AFI + API)."""
    import html as _h
    return sorted({_h.unescape(urllib.parse.unquote(m)) for m in re.findall(r"/imppat/(?:afi|api)formulationdetails/([^\"']+)", home_html)})


def formulation_links(home_html) -> list[tuple[str, str]]:
    """(kind, name) for every formulation page linked from the home page; kind is 'afi' (Ayurvedic Formulary) or 'api' (Pharmacopoeia)."""
    import html as _h
    found = re.findall(r"/imppat/(afi|api)formulationdetails/([^\"']+)", home_html)
    return sorted({(k, _h.unescape(urllib.parse.unquote(n))) for k, n in found})
