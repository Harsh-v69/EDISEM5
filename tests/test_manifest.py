from ayurveda_kg import manifest


def test_add_entry_records_checksum_and_metadata(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    m = tmp_path / "manifest.json"
    e = manifest.add_entry(m, "a", f, "http://x/a", "CC0")
    assert e["sha256"] == manifest.sha256_file(f)
    assert manifest.load(m)["a"]["licence"] == "CC0"
    assert manifest.load(m)["a"]["source_url"] == "http://x/a"


def test_verify_clean_then_detects_change_and_missing(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    g = tmp_path / "b.txt"
    g.write_text("world")
    m = tmp_path / "manifest.json"
    manifest.add_entry(m, "a", f, "u", "l")
    manifest.add_entry(m, "b", g, "u", "l")
    assert manifest.verify(m) == []
    f.write_text("tampered")
    g.unlink()
    assert sorted(manifest.verify(m)) == ["a", "b"]


def test_load_missing_manifest_is_empty(tmp_path):
    assert manifest.load(tmp_path / "nope.json") == {}
