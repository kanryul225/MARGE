"""Integration: discover ML tools from the in-process FastMCP server.

This test boots the real `ml-models` FastMCP server in-process, opens an
MCP client session against it, and verifies that BeeAI's `MCPTool.from_client`
returns one BeeAI Tool per registered MLModel.

Requires the trained model artifacts to exist:
  uv run python -m packages.ml_training.train_breast_cancer
  uv run python -m packages.ml_training.train_diabetes
"""

import pytest

from apps.orchestrator.mcp_discovery import discover_ml_mcp_tools


@pytest.mark.asyncio
async def test_discovers_at_least_one_tool():
    tools = await discover_ml_mcp_tools()
    assert len(tools) >= 1


@pytest.mark.asyncio
async def test_discovered_tools_include_breast_cancer():
    tools = await discover_ml_mcp_tools()
    names = {t.name for t in tools}
    assert "predict_breast_cancer_malignancy" in names


@pytest.mark.asyncio
async def test_discovered_tools_include_diabetes():
    tools = await discover_ml_mcp_tools()
    names = {t.name for t in tools}
    assert "predict_diabetes_risk" in names


@pytest.mark.asyncio
async def test_discovered_tools_are_beeai_mcp_tools():
    from beeai_framework.tools.mcp import MCPTool

    tools = await discover_ml_mcp_tools()
    for t in tools:
        assert isinstance(t, MCPTool)
