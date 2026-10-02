from ayurveda_kg.ingest.crawl_formulations import safe_text


def test_safe_text_is_always_console_encodable_and_keeps_information():
    s = safe_text("FAILED afi Abhayā vaṭī: HTTPError(500)")
    s.encode("cp1252")                                    # must not raise on a Windows console
    assert "Abhay" in s and "HTTPError(500)" in s and r"\u0101" in s
    assert safe_text("plain ascii") == "plain ascii"
