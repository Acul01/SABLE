"""Extractor and checker.

The extractor sees the retrieved passages and must quote the sentences it uses.
The checker sees the claim, the proposed label, the rationale, and those quotes.
It does not see the passages.
"""

from __future__ import annotations

from scripts.sentence_shap import SYSTEM
from sable.schemas import Extraction, Passage, Review
from sable.tools import complete_json, make_llm, parse_scored, passages_text


def label_definitions() -> str:
    start = SYSTEM.index("supported:")
    end = SYSTEM.index("Use only these passages.")
    return SYSTEM[start:end].strip()


EXTRACTOR_SYSTEM = f"""You check one claim against the passages from the same paper.
Reply with JSON only, no markdown. Put the label key first:
{{"label": "supported" | "not supported" | "unclear", "rationale": "<one or two sentences>", "quotes": ["<exact substring of a passage>"]}}

{label_definitions()}
Use only these passages.
quotes must be one or two contiguous pieces copied exactly from the passages, the ones you use for the label."""

CHECKER_SYSTEM = f"""You judge a proposed label for a claim.
You see the claim, the proposed label, a rationale, and the sentences that rationale quotes.
You do not see the paper.
Reply with JSON only, no markdown. Put the label key first:
{{"label": "supported" | "not supported" | "unclear", "rationale": "<one or two sentences>"}}

{label_definitions()}
Give your own label from the quoted sentences. If they do not settle the claim, use unclear."""


def extractor_user(statement: str, passages: list[Passage], feedback: str | None = None) -> str:
    text = f"Claim:\n{statement}\n\n{passages_text(passages)}"
    if feedback:
        text += f"\n\nRevision note:\n{feedback}"
    return text


def checker_user(statement: str, label: str, rationale: str, quotes: list[str]) -> str:
    quoted = "\n".join(f"- {quote}" for quote in quotes)
    return (
        f"Claim:\n{statement}\n\n"
        f"Proposed label: {label}\n\n"
        f"Rationale:\n{rationale}\n\n"
        f"Quoted sentences:\n{quoted}"
    )


def extract_claim(
    statement: str,
    passages: list[Passage],
    feedback: str | None = None,
    *,
    llm=None,
) -> Extraction:
    model = llm if llm is not None else make_llm()
    return complete_json(
        model,
        EXTRACTOR_SYSTEM,
        extractor_user(statement, passages, feedback),
        lambda message: parse_scored(message, Extraction),
    )


def review_claim(
    statement: str,
    label: str,
    rationale: str,
    quotes: list[str],
    *,
    llm=None,
) -> Review:
    model = llm if llm is not None else make_llm()
    return complete_json(
        model,
        CHECKER_SYSTEM,
        checker_user(statement, label, rationale, quotes),
        lambda message: parse_scored(message, Review),
    )
