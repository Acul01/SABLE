"""Schemas for the agent loop. The label set is the version-1 set."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field


def _nonempty(value: str) -> str:
    if not value.strip():
        raise ValueError("empty text")
    return value


Text = Annotated[str, AfterValidator(_nonempty)]
Label = Literal["supported", "not supported", "unclear"]


class Passage(BaseModel):
    index: int
    text: str


class QuoteHit(BaseModel):
    quote: str
    found: bool
    passage_index: int | None = None


class DeleteResult(BaseModel):
    passages: list[Passage]
    removed: int


class LabelScore(BaseModel):
    """Version-1 label. token_probability is the first token of that label."""

    label: Label
    rationale: Text
    token_probability: float


class Extraction(BaseModel):
    label: Label
    rationale: Text
    quotes: list[Text] = Field(min_length=1, max_length=2)
    token_probability: float


class Review(BaseModel):
    label: Label
    rationale: Text
    token_probability: float


class Claim(BaseModel):
    claim_id: str
    doi: str
    statement: str


class Step(BaseModel):
    claim_id: str
    round: int
    tool: str
    output: dict


class ClaimResult(BaseModel):
    claim_id: str
    decision: Literal["accept", "abstain"]
    label: Label | None
    rounds: int
    extractor_label: Label | None
    checker_label: Label | None
    quotes_ok: bool
    token_probability: float | None
    steps: list[Step] = Field(default_factory=list)
