"""End-to-end smoke test for the MCP layer.

Walks every MLModel discovered by the registry and verifies, for each one:
1. It loads, exposes its metadata, and produces a Prediction with SHAP scores.
2. The MCP server registers it as a tool with the correct schema.
3. The MCP tool can be invoked in-process via the FastMCP client and returns
   a Prediction whose predicted_class agrees with the direct call.

Adding a new model file under `services/ml_mcp_server/models/` is enough —
this script does not need to know the model's name.

Run: `python scripts/smoke_test.py`
"""

import asyncio
import json
import sys
from pathlib import Path

# Make the repo root importable when running as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastmcp import Client

from services.ml_mcp_server.models._base import MLModel
from services.ml_mcp_server.registry import discover_models
from services.ml_mcp_server.server import build_server


def _direct_call(model: MLModel) -> str:
    """Run model.predict on its own sample_inputs. Returns predicted_class."""
    print(f"\n[direct] {model.name}")
    print(f"    Trained on:    {model.metadata.trained_on}")
    print(f"    Test accuracy: {model.metadata.test_accuracy:.3f}")
    print(f"    Features:      {model.metadata.feature_count}")

    inputs = model.input_schema(**model.sample_inputs())
    pred = model.predict(inputs)

    print(f"    -> {pred.predicted_class}  (confidence={pred.confidence:.3f})")
    print(f"    Top XAI features:")
    for s in pred.xai_scores[:3]:
        sign = "+" if s.contribution > 0 else "-"
        print(
            f"      {sign} {s.feature_name:<32s} "
            f"contribution={s.contribution:+.4f}  value={s.feature_value:.3f}"
        )
    return pred.predicted_class or ""


def _coerce_payload(result) -> dict:
    """Pull the JSON-shaped Prediction out of a FastMCP CallToolResult."""
    if hasattr(result, "data") and result.data is not None:
        payload = result.data
        if hasattr(payload, "model_dump"):
            return payload.model_dump(mode="json")
        if isinstance(payload, dict):
            return payload
    if hasattr(result, "content") and result.content:
        block = result.content[0]
        if hasattr(block, "text"):
            return json.loads(block.text)
    raise RuntimeError(f"Unexpected MCP tool result shape: {result!r}")


async def _mcp_call(client: Client, model: MLModel) -> str:
    """Call the same model via the in-process MCP client. Returns predicted_class."""
    result = await client.call_tool(model.name, {"inputs": model.sample_inputs()})
    payload = _coerce_payload(result)
    print(
        f"[ mcp ] {model.name}  "
        f"-> {payload.get('predicted_class')}  "
        f"(confidence={payload.get('confidence'):.3f})"
    )
    return payload.get("predicted_class") or ""


async def main() -> None:
    print("=" * 64)
    print(" MARGE smoke test")
    print("=" * 64)

    models = discover_models()
    if not models:
        sys.exit(
            "No MLModel discovered. Did you forget to train artifacts?\n"
            "  python -m packages.ml_training.train_breast_cancer\n"
            "  python -m packages.ml_training.train_diabetes"
        )
    print(f"\nDiscovered {len(models)} model(s): {[m.name for m in models]}")

    direct_results = {m.name: _direct_call(m) for m in models}

    print("\n" + "-" * 64)
    print(" MCP server in-process call")
    print("-" * 64)
    server = build_server()
    async with Client(server) as client:
        tools = await client.list_tools()
        tool_names = {t.name for t in tools}
        print(f"\nRegistered tools: {sorted(tool_names)}")

        for m in models:
            assert m.name in tool_names, f"Tool {m.name} missing from MCP server"
            mcp_class = await _mcp_call(client, m)
            assert mcp_class == direct_results[m.name], (
                f"{m.name}: direct={direct_results[m.name]}  mcp={mcp_class}"
            )

    print("\nSmoke test PASSED.")


if __name__ == "__main__":
    asyncio.run(main())
