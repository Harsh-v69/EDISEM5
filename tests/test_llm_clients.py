import pytest

from ayurveda_kg.rag import generate as G


def boom(p):
    raise RuntimeError("down")


def test_fallback_uses_the_first_working_client_and_reports_which_and_why():
    f = G.with_fallback(boom, lambda p: "local answer")
    assert f("q") == "local answer" and f.used == 1 and f.errors == ["client 0: RuntimeError"]
    g = G.with_fallback(lambda p: "primary", lambda p: "local")
    assert g("q") == "primary" and g.used == 0 and g.errors == []


def test_fallback_treats_an_empty_answer_as_a_failure_and_raises_when_all_fail():
    assert G.with_fallback(lambda p: "  ", lambda p: "ok")("q") == "ok"
    with pytest.raises(RuntimeError):
        G.with_fallback(boom, boom)("q")
    with pytest.raises(RuntimeError):
        G.with_fallback(None)("q")


def test_gemini_client_is_none_without_a_key_and_sends_the_key_in_a_header_not_the_url(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert G.gemini_text_client() is None
    monkeypatch.setenv("GEMINI_API_KEY", "secret-test-key")
    seen = {}

    class R:
        def raise_for_status(self): pass
        def json(self): return {"candidates": [{"content": {"parts": [{"text": "hi "}, {"text": "there"}]}}]}

    import requests
    monkeypatch.setattr(requests, "post", lambda url, **kw: seen.update(url=url, **kw) or R())
    assert G.gemini_text_client()("prompt") == "hi there"
    assert "secret-test-key" not in seen["url"] and seen["headers"]["x-goog-api-key"] == "secret-test-key"


def test_load_env_reads_values_but_never_overrides_existing_variables(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("# comment\nA_TEST_KEY=abc\nB_TEST_KEY='quoted'\n", encoding="utf-8")
    monkeypatch.setenv("B_TEST_KEY", "already")
    monkeypatch.delenv("A_TEST_KEY", raising=False)
    G.load_env(str(p))
    import os
    assert os.environ["A_TEST_KEY"] == "abc" and os.environ["B_TEST_KEY"] == "already"
    monkeypatch.delenv("A_TEST_KEY", raising=False)
