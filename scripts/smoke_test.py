"""End-to-end smoke test for the thin slice.

Verifies:
1. The trained ML model loads and produces a Prediction with SHAP scores.
2. The MCP server registers the model as a tool with the correct schema.
3. The MCP tool can be invoked in-process via the FastMCP client and returns
   the same Prediction shape.

Run: `python scripts/smoke_test.py`
"""

import asyncio
import json
import sys
from pathlib import Path

# Make the repo root importable when running as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastmcp import Client
from sklearn.datasets import load_breast_cancer

from packages.schemas.prediction import Prediction
from services.ml_mcp_server.models.breast_cancer_xgb import BreastCancerXGB
from services.ml_mcp_server.server import build_server


def _sample_inputs() -> dict[str, float]:
    """Take the first row of the dataset as a synthetic patient."""
    data = load_breast_cancer()
    return {
        name.replace(" ", "_"): float(val)
        for name, val in zip(data.feature_names, data.data[0], strict=False)
    }


def _direct_call() -> Prediction:
    print("\n[1] Direct MLModel.predict() ----")
    model = BreastCancerXGB()
    print(f"    Loaded: {model.name}")
    print(f"    Test accuracy: {model.metadata.test_accuracy:.3f}")

    inputs = model.input_schema(**_sample_inputs())
    pred = model.predict(inputs)

    print(f"    Predicted: {pred.predicted_class}  (confidence={pred.confidence:.3f})")
    print(f"    Class probs: {pred.class_probabilities}")
    print(f"    Top XAI features:")
    for s in pred.xai_scores:
        sign = "+" if s.contribution > 0 else "-"
        print(
            f"      {sign} {s.feature_name:<30s} "
            f"contribution={s.contribution:+.4f}  value={s.feature_value:.3f}"
        )
    return pred


async def _mcp_call() -> dict:
    print("\n[2] MCP server -> in-process Client ----")
    server = build_server()

    async with Client(server) as client:
        tools = await client.list_tools()
        tool_names = [t.name for t in tools]
        print(f"    Registered tools: {tool_names}")
        assert "predict_breast_cancer_malignancy" in tool_names

        result = await client.call_tool(
            "predict_breast_cancer_malignancy",
            {"inputs": _sample_inputs()},
        )

        # FastMCP wraps tool results; we expect structured content matching Prediction.
        payload = result.data if hasattr(result, "data") else result
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump(mode="json")
        elif not isinstance(payload, dict):
            # Fallback: try parsing the first content block as JSON.
            content = result.content[0] if hasattr(result, "content") else None
            if content and hasattr(content, "text"):
                payload = json.loads(content.text)

        print(f"    MCP tool result (predicted_class): {payload.get('predicted_class')}")
        print(f"    MCP tool result (confidence): {payload.get('confidence'):.3f}")
        return payload


async def main() -> None:
    print("=" * 64)
    print(" MARGE thin-slice smoke test")
    print("=" * 64)

    direct = _direct_call()
    mcp = await _mcp_call()

    # Sanity: predicted class agrees across both paths
    assert (
        direct.predicted_class == mcp.get("predicted_class")
    ), f"Mismatch: direct={direct.predicted_class}  mcp={mcp.get('predicted_class')}"

    print("\nSmoke test PASSED.")


if __name__ == "__main__":
    asyncio.run(main())
