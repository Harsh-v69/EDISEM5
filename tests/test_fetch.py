import pytest
import requests

from ayurveda_kg import manifest
from ayurveda_kg.ingest.fetch import fetch


class FakeResp:
    def __init__(self, content=b"data", status=200):
        self.content, self.status_code = content, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class FakeSession:
    def __init__(self, resp):
        self.resp, self.calls = resp, []

    def get(self, url, **kw):
        self.calls.append((url, kw))
        return self.resp


def test_fetch_writes_file_and_registers_manifest(tmp_path):
    s = FakeSession(FakeResp(b"abc"))
    m = tmp_path / "m.json"
    p = fetch("http://x/f", tmp_path / "raw" / "f.bin", name="f", licence="CC0",
              manifest_path=m, delay=0, session=s)
    assert p.read_bytes() == b"abc"
    assert manifest.load(m)["f"]["source_url"] == "http://x/f"
    assert "ayurveda-kg" in s.calls[0][1]["headers"]["User-Agent"]


def test_fetch_uses_cache_on_second_call(tmp_path):
    s = FakeSession(FakeResp(b"abc"))
    kw = dict(name="f", licence="CC0", manifest_path=tmp_path / "m.json", delay=0, session=s)
    fetch("http://x/f", tmp_path / "f.bin", **kw)
    fetch("http://x/f", tmp_path / "f.bin", **kw)
    assert len(s.calls) == 1


def test_fetch_http_error_writes_nothing(tmp_path):
    s = FakeSession(FakeResp(status=404))
    dest = tmp_path / "f.bin"
    with pytest.raises(requests.HTTPError):
        fetch("http://x/f", dest, name="f", licence="CC0",
              manifest_path=tmp_path / "m.json", delay=0, session=s)
    assert not dest.exists()
    assert manifest.load(tmp_path / "m.json") == {}


def test_fetch_without_manifest_skips_registration(tmp_path):
    s = FakeSession(FakeResp(b"abc"))
    p = fetch("http://x/f", tmp_path / "f.bin", name="f", licence="CC0", manifest_path=None, delay=0, session=s)
    assert p.read_bytes() == b"abc"


def test_fetch_refetches_empty_cached_file(tmp_path):
    dest = tmp_path / "f.bin"
    dest.write_bytes(b"")                       # a crashed/empty earlier download must not count as cached
    s = FakeSession(FakeResp(b"abc"))
    fetch("http://x/f", dest, name="f", licence="CC0", manifest_path=None, delay=0, session=s)
    assert dest.read_bytes() == b"abc" and len(s.calls) == 1
