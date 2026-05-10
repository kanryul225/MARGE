"""Tests for the medical expert web search tool."""

import tavily

from services.medical_expert_agent.tools.search_web import search_web


def test_search_web_returns_empty_when_tavily_key_missing(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("MEDICAL_WEB_SEARCH_API_KEY", raising=False)

    assert search_web("diabetes clinical guidance") == []


def test_search_web_filters_to_trusted_medical_domains_by_default(monkeypatch):
    calls = {}

    class FakeTavilyClient:
        def __init__(self, api_key):
            calls["api_key"] = api_key

        def search(self, **kwargs):
            calls.update(kwargs)
            return {"results": []}

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.delenv("MEDICAL_WEB_SEARCH_INCLUDE_DOMAINS", raising=False)
    monkeypatch.setattr(tavily, "TavilyClient", FakeTavilyClient)

    assert search_web("diabetes clinical guidance") == []
    assert calls["include_domains"] == ["medlineplus.gov", "pubmed.ncbi.nlm.nih.gov"]


def test_search_web_uses_configured_domain_filter(monkeypatch):
    calls = {}

    class FakeTavilyClient:
        def __init__(self, api_key):
            calls["api_key"] = api_key

        def search(self, **kwargs):
            calls.update(kwargs)
            return {"results": []}

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("MEDICAL_WEB_SEARCH_INCLUDE_DOMAINS", "medlineplus.gov, cdc.gov")
    monkeypatch.setattr(tavily, "TavilyClient", FakeTavilyClient)

    assert search_web("diabetes clinical guidance") == []
    assert calls["include_domains"] == ["medlineplus.gov", "cdc.gov"]
