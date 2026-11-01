"""One claim at a time, in plain Python.

The loop retrieves, asks the extractor, checks that the quotes are verbatim,
and asks the checker. The checker does not receive the passages. Agreement
and verbatim quotes accept the label. Otherwise the extractor gets one
revision note and the loop runs once more. The second miss abstains.

delete_span and score_label stay available for a later comparison with
version 1. This loop does not call them: a deletion inside the loop would
mix the checker with the intervention.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from sable.agents import extract_claim, review_claim
from sable.schemas import Claim, ClaimResult, Extraction, QuoteHit, Review, Step
from sable.tools import ROOT, check_quotes, retrieve_passages


def revision_feedback(extraction: Extraction, review: Review, hits: list[QuoteHit]) -> str:
    missing = [hit.quote for hit in hits if not hit.found]
    lines = [
        "A second reader saw only your label, rationale, and quotes, not the passages.",
        f"Your label was {extraction.label}. The second reader labeled the claim {review.label}.",
        f"Second reader: {review.rationale}",
    ]
    if missing:
        lines.append("These quotes are not exact substrings of the passages:")
        lines.extend(f"- {quote}" for quote in missing)
    else:
        lines.append("The quotes are exact substrings of the passages.")
    lines.append("Label the claim again from the passages. Put the label key first.")
    return "\n".join(lines)


def load_claim(claim_id: str) -> Claim:
    with (ROOT / "data" / "CLAIMS.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["claim_id"] == claim_id:
                return Claim(
                    claim_id=row["claim_id"],
                    doi=row["doi"],
                    statement=row["statement"],
                )
    raise KeyError(claim_id)


def _append(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_claim(
    claim: Claim,
    *,
    retrieve=retrieve_passages,
    extract=extract_claim,
    check=check_quotes,
    review=review_claim,
    log_path: Path | None = None,
    max_rounds: int = 2,
) -> ClaimResult:
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    steps: list[Step] = []

    def record(round_index: int, tool: str, output: dict) -> None:
        step = Step(claim_id=claim.claim_id, round=round_index, tool=tool, output=output)
        steps.append(step)
        if log_path is not None:
            _append(log_path, step.model_dump(mode="json"))

    passages = retrieve(claim.statement, claim.doi)
    record(0, "retrieve", {
        "doi": claim.doi,
        "n": len(passages),
        "passages": [passage.model_dump() for passage in passages],
    })

    feedback = None
    extraction = None
    checker = None
    quotes_ok = False
    rounds = 0
    for round_index in range(1, max_rounds + 1):
        if not passages:
            break
        rounds = round_index
        extraction = extract(claim.statement, passages, feedback)
        record(round_index, "extract", extraction.model_dump(mode="json"))
        hits = check(extraction.quotes, passages)
        quotes_ok = bool(hits) and all(hit.found for hit in hits)
        record(round_index, "check_quotes", {"hits": [hit.model_dump() for hit in hits]})
        checker = review(
            claim.statement,
            extraction.label,
            extraction.rationale,
            extraction.quotes,
        )
        record(round_index, "review", checker.model_dump(mode="json"))
        if quotes_ok and checker.label == extraction.label:
            record(round_index, "decision", _decision_payload(
                "accept", extraction, checker, quotes_ok, rounds,
            ))
            return _result(claim, "accept", extraction, checker, quotes_ok, rounds, steps)
        feedback = revision_feedback(extraction, checker, hits)

    record(rounds, "decision", _decision_payload(
        "abstain", extraction, checker, quotes_ok, rounds,
    ))
    return _result(claim, "abstain", extraction, checker, quotes_ok, rounds, steps)


def _result(claim, decision, extraction, checker, quotes_ok, rounds, steps) -> ClaimResult:
    return ClaimResult(
        claim_id=claim.claim_id,
        decision=decision,
        label=extraction.label if decision == "accept" and extraction is not None else None,
        rounds=rounds,
        extractor_label=None if extraction is None else extraction.label,
        checker_label=None if checker is None else checker.label,
        quotes_ok=quotes_ok,
        token_probability=None if extraction is None else extraction.token_probability,
        steps=steps,
    )


def _decision_payload(decision, extraction, checker, quotes_ok, rounds) -> dict:
    accepted = decision == "accept" and extraction is not None
    return {
        "decision": decision,
        "label": extraction.label if accepted else None,
        "rounds": rounds,
        "extractor_label": None if extraction is None else extraction.label,
        "checker_label": None if checker is None else checker.label,
        "quotes_ok": quotes_ok,
        "token_probability": None if extraction is None else extraction.token_probability,
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    if len(argv) != 2:
        print("usage: python -m sable.orchestrator CLAIM_ID", file=sys.stderr)
        return 2
    claim = load_claim(argv[1])
    path = ROOT / "results" / "v2" / "trace.jsonl"
    result = run_claim(claim, log_path=path)
    print(result.model_dump_json(indent=2, exclude={"steps"}))
    print(f"log: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
