"""Discover, validate and load project agents and skills into Claude Code.

Source of truth: ``.agents/agents/*.md`` and ``.agents/skills/*/SKILL.md``.
Claude Code natively loads subagents from ``.claude/agents`` and skills from
``.claude/skills``; this script mirrors the sources there (copy, so it works on
Windows without symlink privileges) after validating them.

Usage::

    python scripts/sync_agents.py            # validate + mirror
    python scripts/sync_agents.py --check    # validate only (exit 1 on problems)
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENTS_SRC = ROOT / ".agents" / "agents"
SKILLS_SRC = ROOT / ".agents" / "skills"
AGENTS_DST = ROOT / ".claude" / "agents"
SKILLS_DST = ROOT / ".claude" / "skills"

REQUIRED_SKILL_SECTIONS = [
    "Purpose",
    "Activation conditions",
    "Required inputs",
    "Execution procedure",
    "Constraints",
    "Expected outputs",
    "Validation checklist",
]


@dataclass
class Skill:
    name: str
    description: str
    path: Path


@dataclass
class AgentDef:
    name: str
    description: str
    skills: list[str]
    path: Path


def _frontmatter(text: str) -> dict[str, str]:
    m = re.match(r"^---\n(.*?)\n---\n", text.replace("\r\n", "\n"), re.S)
    if not m:
        return {}
    out: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def discover_skills() -> tuple[dict[str, Skill], list[str]]:
    skills: dict[str, Skill] = {}
    problems: list[str] = []
    for skill_md in sorted(SKILLS_SRC.glob("*/SKILL.md")):
        text = skill_md.read_text(encoding="utf-8")
        fm = _frontmatter(text)
        name = fm.get("name", "")
        if name != skill_md.parent.name:
            problems.append(f"{skill_md}: frontmatter name '{name}' != directory '{skill_md.parent.name}'")
        if not fm.get("description"):
            problems.append(f"{skill_md}: missing description")
        for section in REQUIRED_SKILL_SECTIONS:
            if not re.search(rf"^## {re.escape(section)}\s*$", text, re.M):
                problems.append(f"{skill_md}: missing section '## {section}'")
        skills[skill_md.parent.name] = Skill(name, fm.get("description", ""), skill_md)
    return skills, problems


def discover_agents(skills: dict[str, Skill]) -> tuple[dict[str, AgentDef], list[str]]:
    agents: dict[str, AgentDef] = {}
    problems: list[str] = []
    for md in sorted(AGENTS_SRC.glob("*.md")):
        fm = _frontmatter(md.read_text(encoding="utf-8"))
        name = fm.get("name", "")
        if name != md.stem:
            problems.append(f"{md}: frontmatter name '{name}' != file name '{md.stem}'")
        raw = fm.get("skills", "").strip("[]")
        skill_names = [s.strip() for s in raw.split(",") if s.strip()]
        if not skill_names:
            problems.append(f"{md}: declares no skills")
        for s in skill_names:
            if s not in skills:
                problems.append(f"{md}: references unknown skill '{s}'")
        agents[md.stem] = AgentDef(name, fm.get("description", ""), skill_names, md)
    return agents, problems


def mirror(skills: dict[str, Skill], agents: dict[str, AgentDef]) -> None:
    for dst in (AGENTS_DST, SKILLS_DST):
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
    for a in agents.values():
        shutil.copy2(a.path, AGENTS_DST / a.path.name)
    for s in skills.values():
        shutil.copytree(s.path.parent, SKILLS_DST / s.path.parent.name)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="validate only")
    args = ap.parse_args(argv)
    skills, p1 = discover_skills()
    agents, p2 = discover_agents(skills)
    problems = p1 + p2
    print(f"skills: {len(skills)}  agents: {len(agents)}")
    for p in problems:
        print("PROBLEM:", p)
    if problems:
        return 1
    if not args.check:
        mirror(skills, agents)
        print(f"mirrored to {AGENTS_DST.relative_to(ROOT)} and {SKILLS_DST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
