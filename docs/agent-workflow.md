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
| T1 | Data engine | data-engineering | todo | |
| T2 | Runtime & launch | runtime-compatibility | todo | |
| T3 | Security module | security | todo | |
| T4 | Frontend | frontend-engineering | todo | |
| T5 | Monitoring | gpu-memory-optimization | todo | |
| T6 | Training & export | llm-training | todo | |
| T7 | Backend API | backend-engineering | todo | |
| T8 | Studio facade | chief-architect | todo | |
| T9 | QA | qa-testing | todo | |
| T10 | Security review | security | todo | |
| T11 | Docs & notebooks | documentation | todo | |
| T12 | Final review | chief-architect | todo | |

## Review protocol
Each agent returns a report (files, skills loaded, tests + output, UNVERIFIED items, out-of-scope requests). The Architect re-runs the agent's tests, reviews the diff against the skills' checklists, then commits the unit (`git commit` per task). Conflicts are resolved by the Architect editing contracts (with an ADR) and re-dispatching.
