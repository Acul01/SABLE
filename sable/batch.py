"""Run the loop on every claim and write the comparison with version 1.

A claim already recorded with a decision in the log is skipped, so a stopped
run continues from the next claim. Version 2a stops after one round. Version
2b allows the one revision already built into the loop.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from sable.orchestrator import load_claim, run_claim
from sable.tools import ROOT, ensure_env, make_llm, retrieve_passages

V2 = ROOT / "results" / "v2"
CONDITIONS = {
    "v2a": 1,
    "v2b": 2,
}


def load_claims() -> list:
    with (ROOT / "data" / "CLAIMS.csv").open(encoding="utf-8") as handle:
        return [load_claim(row["claim_id"]) for row in csv.DictReader(handle)]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def finished_ids(path: Path) -> set[str]:
    return {row["claim_id"] for row in read_jsonl(path) if row.get("tool") == "decision"}


def run_condition(name: str, log_path: Path | None = None) -> Path:
    from scripts.sentence_shap import open_store

    log_path = V2 / f"{name}.jsonl" if log_path is None else log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_env()
    store, k = open_store()
    llm = make_llm()

    def retrieve(statement: str, doi: str):
        return retrieve_passages(statement, doi, k=k, store=store)

    def extract(statement: str, passages, feedback=None):
        from sable.agents import extract_claim
        return extract_claim(statement, passages, feedback, llm=llm)

    def review(statement: str, label: str, rationale: str, quotes: list[str]):
        from sable.agents import review_claim
        return review_claim(statement, label, rationale, quotes, llm=llm)

    claims = load_claims()
    done = finished_ids(log_path)
    max_rounds = CONDITIONS[name]
    print(f"{name}: {len(done)}/{len(claims)} already finished", flush=True)
    for index, claim in enumerate(claims, start=1):
        if claim.claim_id in done:
            continue
        print(f"{name} {index}/{len(claims)} {claim.claim_id}", flush=True)
        try:
            result = run_claim(
                claim,
                retrieve=retrieve,
                extract=extract,
                review=review,
                log_path=log_path,
                max_rounds=max_rounds,
            )
        except Exception as error:
            print(f"{name} {claim.claim_id} error: {error}", flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "claim_id": claim.claim_id,
                    "round": 0,
                    "tool": "error",
                    "output": {"error": str(error)},
                }) + "\n")
            continue
        print(
            f"  {result.decision} {result.label or '-'} rounds={result.rounds}",
            flush=True,
        )
    return log_path


def load_v1() -> dict[str, dict]:
    with (ROOT / "results" / "predictions.csv").open(encoding="utf-8") as handle:
        return {row["claim_id"]: row for row in csv.DictReader(handle)}


def claim_table(name: str, log_path: Path) -> list[dict]:
    v1 = load_v1()
    decisions = {}
    for row in read_jsonl(log_path):
        if row.get("tool") == "decision":
            decisions[row["claim_id"]] = row
    table = []
    for claim_id, baseline in v1.items():
        decision = decisions.get(claim_id)
        output = {} if decision is None else decision["output"]
        label = output.get("label") or ""
        accepted = output.get("decision") == "accept"
        table.append({
            "condition": name,
            "claim_id": claim_id,
            "gold": baseline["gold"],
            "v1_label": baseline["predicted"],
            "v1_match": str(baseline["predicted"] == baseline["gold"]),
            "evidence_in_topk": baseline["evidence_in_topk"],
            "decision": output.get("decision", ""),
            "label": label,
            "match": str(label == baseline["gold"]) if accepted else "",
            "rounds": output.get("rounds", ""),
            "extractor_label": output.get("extractor_label") or "",
            "checker_label": output.get("checker_label") or "",
            "quotes_ok": output.get("quotes_ok", ""),
            "token_probability": output.get("token_probability", ""),
        })
    return table


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def condition_summary(name: str, rows: list[dict], rescued: set[str] | None = None) -> dict:
    present = [row for row in rows if row["decision"]]
    accepted = [row for row in present if row["decision"] == "accept"]
    correct = [row for row in accepted if row["match"] == "True"]
    disagreements = [
        row for row in present
        if row["extractor_label"] and row["checker_label"] and row["extractor_label"] != row["checker_label"]
    ]
    quote_failures = [row for row in present if row["quotes_ok"] is False or row["quotes_ok"] == "False"]
    rounds = [int(row["rounds"]) for row in present if row["rounds"] != ""]
    rescued_rows = [row for row in rows if rescued and row["claim_id"] in rescued and row["decision"] == "accept"]
    return {
        "condition": name,
        "claims": len(rows),
        "finished": len(present),
        "accepted": len(accepted),
        "abstained": sum(row["decision"] == "abstain" for row in present),
        "correct_accepts": len(correct),
        "wrong_accepts": len(accepted) - len(correct),
        "v1_correct": sum(row["v1_match"] == "True" for row in rows),
        "checker_disagreements": len(disagreements),
        "quotes_failed": len(quote_failures),
        "mean_rounds": round(sum(rounds) / len(rounds), 3) if rounds else "",
        "rescued_by_second_round": len(rescued_rows),
        "rescued_and_correct": sum(row["match"] == "True" for row in rescued_rows),
    }


def write_tables() -> Path:
    tables = {}
    for name in CONDITIONS:
        rows = claim_table(name, V2 / f"{name}.jsonl")
        write_csv(V2 / f"{name}.csv", rows)
        tables[name] = rows
    abstained_a = {row["claim_id"] for row in tables["v2a"] if row["decision"] == "abstain"}
    summary = [
        condition_summary("v2a", tables["v2a"]),
        condition_summary("v2b", tables["v2b"], abstained_a),
    ]
    path = V2 / "comparison.csv"
    write_csv(path, summary)
    return path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    chosen = argv or ["v2a", "v2b"]
    unknown = set(chosen) - set(CONDITIONS)
    if unknown:
        print("usage: python -m sable.batch [v2a] [v2b]", file=sys.stderr)
        return 2
    for name in chosen:
        run_condition(name)
    if set(chosen) == set(CONDITIONS):
        path = write_tables()
        print(path.read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
