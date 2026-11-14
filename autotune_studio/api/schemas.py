"""Small request/response models local to the API layer.

These shapes are not (yet) in :mod:`autotune_studio.contracts`; they are requested for promotion
there in the backend report so ``scripts/gen_ts_types.py`` can emit TypeScript for them.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(_Model):
    status: str = "ok"
    version: str


class SessionRequest(_Model):
    token: str = Field(..., min_length=1, max_length=512)


class OkResponse(_Model):
    ok: bool = True


class StartTrainingRequest(_Model):
    dataset_id: str | None = Field(None, max_length=128)
    resume_from: str | None = Field(None, max_length=256, description="Checkpoint id to continue from.")


class ResumeCheckpointRequest(_Model):
    dataset_id: str | None = Field(None, max_length=128)


class TrainingValidation(_Model):
    planned_steps: int
    warnings: list[str] = Field(default_factory=list)
