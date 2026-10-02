from ayurveda_kg.ingest.audit import SOURCES, probe, write_report


class R:
    def __init__(self, code, ctype="text/html"):
        self.status_code, self.headers = code, {"Content-Type": ctype}

    def close(self):
        pass


class S:
    def __init__(self, r=None, exc=None):
        self.r, self.exc = r, exc

    def get(self, url, **kw):
        if self.exc:
            raise self.exc
        assert kw.get("stream") is True  # must not download the body
        return self.r


def test_probe_ok_and_http_error_and_exception():
    assert probe("u", S(R(200)))["status"] == "OK"
    assert probe("u", S(R(403)))["status"] == "HTTP 403"
    out = probe("u", S(exc=ConnectionError("boom")))
    assert out["status"] == "ERROR" and "boom" in out["note"]


def test_sources_cover_every_planned_dataset():
    names = " ".join(s["name"] for s in SOURCES).lower()
    for must in ["imppat", "pubchem", "chembl", "dgidb", "ddinter", "drugbank", "ayurparam"]:
        assert must in names


def test_write_report(tmp_path):
    p = tmp_path / "r.md"
    write_report([{"name": "X", "url": "u", "needs": "n", "status": "OK", "http": 200, "type": "t", "note": ""}], p)
    txt = p.read_text(encoding="utf-8")
    assert "| X |" in txt and "OK" in txt
