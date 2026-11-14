"""Agent/skill configuration is valid and discoverable (owned by chief-architect)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_sync():
    spec = importlib.util.spec_from_file_location("sync_agents", ROOT / "scripts" / "sync_agents.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    import sys

    sys.modules["sync_agents"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_all_required_skills_exist_and_are_complete():
    sync = _load_sync()
    skills, problems = sync.discover_skills()
    assert problems == []
    assert set(skills) >= {
        "architecture-design", "notebook-runtime-detection", "streaming-data-engineering",
        "dataset-validation", "llm-finetuning", "gpu-memory-optimization", "fastapi-backend",
        "react-dashboard", "security-audit", "testing-and-verification",
    }


def test_all_agents_reference_existing_skills():
    sync = _load_sync()
    skills, _ = sync.discover_skills()
    agents, problems = sync.discover_agents(skills)
    assert problems == []
    assert len(agents) == 10
