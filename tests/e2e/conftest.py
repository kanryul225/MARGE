"""E2E test configuration — stub heavy optional deps so UI logic tests can import app.py."""

import sys
from types import ModuleType
from unittest.mock import MagicMock


def _stub(name: str) -> ModuleType:
    """Install a ModuleType stub where attribute access returns MagicMock()."""
    mod = ModuleType(name)
    mod.__getattr__ = lambda attr: MagicMock()  # type: ignore[method-assign]
    sys.modules[name] = mod
    return mod


def _ensure(name: str) -> MagicMock:
    return sys.modules.setdefault(name, _stub(name))  # type: ignore[return-value]


def _install_stubs() -> None:
    # --- streamlit ---
    _ensure("streamlit")

    # --- dotenv ---
    _ensure("dotenv")

    # --- beeai_framework — stub every sub-path that might be imported ---
    for path in (
        "beeai_framework",
        "beeai_framework.backend",
        "beeai_framework.backend.message",
        "beeai_framework.agents",
        "beeai_framework.agents.requirement",
        "beeai_framework.agents.requirement.requirements",
        "beeai_framework.agents.requirement.requirements.conditional",
        "beeai_framework.tools",
        "beeai_framework.tools.mcp",
        "beeai_framework.emitter",
        "beeai_framework.memory",
    ):
        _ensure(path)

    # Provide real-ish UserMessage so it can be instantiated
    sys.modules["beeai_framework.backend.message"].UserMessage = MagicMock

    # --- apps.orchestrator — stub to avoid transitive BeeAI imports ---
    for path in (
        "apps.orchestrator",
        "apps.orchestrator.agent",
        "apps.orchestrator.requirements",
        "apps.orchestrator.requirements.marge_protocol",
        "apps.orchestrator.mcp_discovery",
        "apps.orchestrator.middleware",
        "apps.orchestrator.middleware.enforce_protocol",
        "apps.orchestrator.tools",
        "apps.orchestrator.tools.consult_expert",
        "apps.orchestrator.tools.patient_history",
        "apps.orchestrator.tools.final_report",
        "apps.orchestrator.tools._adapter",
    ):
        _ensure(path)

    # Provide callable stubs for what app.py actually calls
    orch_agent = sys.modules["apps.orchestrator.agent"]
    orch_agent.build_bundle = MagicMock()
    orch_agent.orchestrator_agent = MagicMock()

    # --- packages.llm_provider — stub LLM client to avoid env-var checks ---
    for path in (
        "packages.llm_provider",
        "packages.llm_provider.client",
        "packages.llm_provider.settings",
    ):
        _ensure(path)

    from packages.llm_provider import settings as _s  # real module is importable; this is fine
    # If the real module already loaded, leave it alone; otherwise keep the stub


_install_stubs()
