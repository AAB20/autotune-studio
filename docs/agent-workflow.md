# Agent workflow, task board and dependency graph

## Dependency graph
```
T0 Architect: contracts, interfaces, errors, events, ADRs, skills, agents
 ├─► T1 data-engineering (connectors, data_engine)
 ├─► T2 runtime-compatibility (environment, hardware, launch)
 ├─► T3 security (security module)
 ├─► T4 frontend-engineering (SPA vs generated types)
 │
 ├─► T5 gpu-memory-optimization (monitoring) ─► T6 llm-training (training, export)  [needs T1]
 ├─► T7 backend-engineering (api, lifecycle, config)                               [needs T3; uses Studio facade]
 │
 T8 Architect: studio.py facade + cli.py                                            [needs T1,T2,T6]
 T9 qa-testing (integration, compatibility, acceptance matrix)                      [needs T8,T7,T4]
 T10 security review                                                                [needs T9]
 T11 documentation + notebooks                                                      [needs T10]
 T12 Architect final integration review + report                                    [needs all]
```

## Task board
| ID | Task | Owner | Status | Evidence |
|---|---|---|---|---|
| T0 | Scaffold & contracts | chief-architect | done | `tests/unit/test_agent_config.py`, `test_imports.py` |
| T1 | Data engine | data-engineering | done | `tests/unit/data` (74) |
| T2 | Runtime & launch | runtime-compatibility | done | `tests/unit/runtime` (76) |
| T3 | Security module | security | done | `tests/unit/security` |
| T4 | Frontend | frontend-engineering | done | vitest 54, `npm run build` |
| T5 | Monitoring | gpu-memory-optimization | done | `tests/unit/monitoring` (49) |
| T6 | Training & export | llm-training | done | `tests/unit/training` (34) |
| T7 | Backend API | backend-engineering | done | `tests/unit/api` (109) |
| T8 | Studio facade | chief-architect | done | `tests/unit/core`, browser E2E |
| T9 | QA | qa-testing | done | `tests/integration`, `tests/compatibility`, `docs/acceptance.md` |
| T10 | Security review | security | done | `docs/security-review.md` SEC-01..12 |
| T11 | Docs & notebooks | documentation | done | README, docs/*, notebooks (validated, unexecuted) |
| T12 | Final review | chief-architect | done | `docs/final-report.md` |

## Review protocol
Each agent returns a report (files, skills loaded, tests + output, UNVERIFIED items, out-of-scope requests). The Architect re-runs the agent's tests, reviews the diff against the skills' checklists, then commits the unit (`git commit` per task). Conflicts are resolved by the Architect editing contracts (with an ADR) and re-dispatching.
