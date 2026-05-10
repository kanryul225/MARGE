"""Expert-only live medical web search tool.

The orchestrator never imports this module. It is wired only inside
`MedicalExpertAgent`, so any `search_medical_web` call in the trace proves the
expert model initiated the retrieval step.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from packages.schemas.retrieval import RetrievedDocument

TOOL_NAME = "search_medical_web"
DEFAULT_INCLUDE_DOMAINS = ("medlineplus.gov", "pubmed.ncbi.nlm.nih.gov")
TOOL_DESCRIPTION = (
    "Search the live web for current, citable medical guidance. Use this before "
    "making guideline, diagnostic-threshold, treatment, or quantitative clinical "
    "claims. Prefer authoritative sources such as ADA, CDC, WHO, NICE, USPSTF, "
    "NIH, and peer-reviewed clinical references."
)


class ToolInput(BaseModel):
    query: str = Field(
        description=(
            "Focused medical search query, including the condition, value, "
            "threshold, or guideline body when relevant."
        )
    )
    max_results: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum number of web results to return.",
    )


def _medical_web_include_domains() -> list[str]:
    raw = os.getenv("MEDICAL_WEB_SEARCH_INCLUDE_DOMAINS")
    if not raw:
        return list(DEFAULT_INCLUDE_DOMAINS)

    domains = [part.strip() for part in raw.replace(";", ",").split(",")]
    return [domain for domain in domains if domain]


def search_web(query: str, max_results: int = 3) -> list[RetrievedDocument]:
    """Run a Tavily-backed web search and return normalized retrieval docs.

    This sync helper is intentionally small and testable. Missing optional
    setup returns an empty list so expert consultation can continue.
    """

    api_key = os.getenv("TAVILY_API_KEY") or os.getenv("MEDICAL_WEB_SEARCH_API_KEY")
    if not api_key:
        return []

    try:
        from tavily import TavilyClient
    except ImportError:
        return []

    client = TavilyClient(api_key=api_key)
    raw = client.search(
        query=query,
        max_results=max_results,
        search_depth="basic",
        include_answer=False,
        include_domains=_medical_web_include_domains(),
    )

    documents: list[RetrievedDocument] = []
    for idx, item in enumerate(raw.get("results", [])[:max_results], start=1):
        score = item.get("score")
        try:
            score_value = float(score) if score is not None else float(idx)
        except (TypeError, ValueError):
            score_value = float(idx)

        doc = RetrievedDocument(
            title=str(item.get("title") or "Untitled medical web result"),
            snippet=str(item.get("content") or item.get("snippet") or ""),
            source_url=item.get("url"),
            retrieval_source="web",
            score=score_value,
        )
        documents.append(doc)

    return documents


async def search_medical_web(query: str, max_results: int = 3) -> dict[str, Any]:
    """Expert-tool wrapper returning JSON-safe payload for BeeAI tracing."""

    include_domains = _medical_web_include_domains()
    documents = search_web(query=query, max_results=max_results)
    warning = None
    if not documents:
        if not (os.getenv("TAVILY_API_KEY") or os.getenv("MEDICAL_WEB_SEARCH_API_KEY")):
            warning = "TAVILY_API_KEY is not set; live web search was not executed."
        else:
            warning = "No web results returned, or tavily-python is not installed."

    return {
        "query": query,
        "include_domains": include_domains,
        "documents": [doc.model_dump(mode="json") for doc in documents],
        "warning": warning,
    }
