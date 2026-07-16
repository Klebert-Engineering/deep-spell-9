"""API response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClassProb(BaseModel):
    name: str
    p: float


class CompletionOut(BaseModel):
    text: str
    class_name: str
    logprob: float


class SuggestionOut(BaseModel):
    text: str
    distance: float


class CorrectionOut(BaseModel):
    input: str
    suggestions: list[SuggestionOut] = Field(default_factory=list)


class CompleteResponse(BaseModel):
    query: str
    classes: list[list[ClassProb]]  # per input character, probability-descending
    completions: list[CompletionOut]
    tokens: dict[str, str]  # best class -> token string (with top completion applied)
    corrections: dict[str, CorrectionOut]
    timings_ms: dict[str, float]


class HealthResponse(BaseModel):
    status: str = "ok"
    with_corrector: bool = False
    with_lookup: bool = False
