"""Phase 1: crawl IMPPAT pages for the scoped herbs (resumable; run in background, ~1h)."""
from ayurveda_kg.ingest.imppat import crawl_compounds, crawl_plants, parse_plant_page, plant_ids
from ayurveda_kg.scope import load_scope

RAW = "data/raw/imppat"
herbs = [h["imppat_name"] for h in load_scope()["herbs"]]
pages = crawl_plants(herbs, RAW)
rows = []
for h, p in pages.items():
    r = parse_plant_page(p.read_text(encoding="utf-8", errors="ignore"))
    print(f"{h:28s} {len(r):4d} associations", flush=True)
    rows += r
ids = plant_ids(rows)
print("unique compounds:", len(ids), flush=True)
print(crawl_compounds(ids, RAW), flush=True)
