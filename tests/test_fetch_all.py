from pathlib import Path

from ayurveda_kg.ingest import fetch_all


def test_the_download_list_covers_every_open_source_with_unique_destinations_and_https_urls():
    files = fetch_all.FILES
    dests = [f["dest"] for f in files]
    assert len(dests) == len(set(dests)) and all(f["url"].startswith("https://") for f in files)
    groups = {Path(d).parts[2] for d in dests}                                      # data/raw/<group>/...
    assert groups == {"dgidb", "ddinter", "tdc_cyp", "texts"}
    assert sum("dgidb" in d for d in dests) == 3 and sum("ddinter" in d for d in dests) == 8
    assert sum("tdc_cyp" in d for d in dests) == 3 and sum("texts" in d for d in dests) == 4
    assert all(f["licence"] and f["name"] for f in files)


def test_main_fetches_every_file_through_the_tracked_fetcher_and_then_runs_the_chembl_step(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(fetch_all, "fetch", lambda url, dest, **kw: calls.append((url, str(dest), kw.get("manifest_path"))) or Path(dest))
    chembl_calls = []
    monkeypatch.setattr(fetch_all, "fetch_chembl", lambda: (chembl_calls.append(1), {"warfarin": "CHEMBL1464"})[1])
    fetch_all.main()
    assert len(calls) == len(fetch_all.FILES) and chembl_calls == [1]
    assert all(m == "data/manifest.json" for _, _, m in calls)                       # everything is recorded with source, licence and checksum
