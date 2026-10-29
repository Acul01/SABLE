"""Five-sample labels for every condition that was previously run once.

Sampling follows the thesis setting: temperature 0.7, top-p 0.9, five draws.
Agreement is the share of draws that repeat the majority label.
Confidence is the mean probability of the decoded label token on the draws
that produced that majority label.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from scripts.sentence_shap import (
    SYSTEM,
    open_store,
    passages_without_evidence,
    select_sentences,
    stayed_supported,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "data" / "sample_cache.jsonl"
SAMPLES_PATH = ROOT / "results" / "sampled_labels.csv"
SUMMARY_PATH = ROOT / "results" / "sampled_summary.csv"
RESULTS_PATH = ROOT / "results" / "results_sampled.csv"
N_SAMPLES = 5


def label_and_confidence(message) -> tuple[str, float]:
    """Parsed label, and the probability of the first token of that label."""
    raw = message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        label = json.loads(raw)["label"]
    except json.JSONDecodeError:
        label = ""
        for name in ("not supported", "supported", "unclear"):
            if f'"{name}"' in raw:
                label = name
                break
    if label not in {"supported", "not supported", "unclear"}:
        raise ValueError(f"unparsed label: {raw[:180]}")
    content = (message.response_metadata.get("logprobs") or {}).get("content") or []
    text = ""
    waiting = False
    for token in content:
        text += token["token"]
        if not waiting and (text.rstrip().endswith('{"label": "') or text.rstrip().endswith('{"label":"')):
            waiting = True
            continue
        if waiting:
            return label, math.exp(token["logprob"])
    raise ValueError(f"label token not found in logprobs: {raw[:180]}")


def aggregate(samples: list[dict]) -> tuple[str, float, float]:
    labels = [sample["label"] for sample in samples]
    counts = Counter(labels)
    top = max(counts.values())
    tied = {label for label, count in counts.items() if count == top}
    majority = next(label for label in labels if label in tied)
    agreement = top / len(labels)
    chosen = [sample["confidence"] for sample in samples if sample["label"] == majority]
    confidence = sum(chosen) / len(chosen)
    return majority, agreement, confidence


def load_done() -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    if not CACHE_PATH.exists():
        return grouped
    for line in CACHE_PATH.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        grouped.setdefault((row["condition"], row["claim_id"]), []).append(row)
    return grouped


def append_sample(row: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def prompt_for(statement: str, parts: list[str]) -> str:
    return f"Claim:\n{statement}\n\n" + "\n\n".join(parts)


def chunk_parts(docs, evidence: str | None) -> list[str]:
    parts = []
    for number, doc in enumerate(docs, start=1):
        text = doc.page_content
        if evidence is not None and evidence in text:
            text = text.replace(evidence, " ")
        parts.append(f"Passage {number}:\n{text}")
    return parts


def sentence_parts(docs, evidence: str, drop: set[tuple[int, str]]) -> list[str]:
    passages = passages_without_evidence(docs, evidence)
    parts = []
    for number, sentences in enumerate(passages, start=1):
        kept = [sentence for sentence in sentences if (number, sentence) not in drop]
        body = " ".join(kept) if kept else "(no text)"
        parts.append(f"Passage {number}:\n{body}")
    return parts


def build_jobs() -> list[dict]:
    claims = pd.read_csv(ROOT / "data" / "CLAIMS.csv").set_index("claim_id")
    predictions = pd.read_csv(ROOT / "results" / "predictions.csv")
    shap_frame = pd.read_csv(ROOT / "results" / "shap_sentences.csv")
    removals = pd.read_csv(ROOT / "results" / "shap_removal.csv")
    if "selection" not in removals.columns:
        removals["selection"] = "highest"
    store, k = open_store()
    docs_by_claim: dict[str, list] = {}

    def docs_for(claim_id: str, doi: str):
        if claim_id not in docs_by_claim:
            docs_by_claim[claim_id] = store.similarity_search(
                claims.loc[claim_id, "statement"],
                k=k,
                filter={"source": doi},
            )
        return docs_by_claim[claim_id]

    jobs = []
    for _, row in predictions.iterrows():
        claim_id = row["claim_id"]
        statement = claims.loc[claim_id, "statement"]
        evidence = claims.loc[claim_id, "evidence"]
        docs = docs_for(claim_id, row["doi"])
        jobs.append({
            "condition": "original",
            "claim_id": claim_id,
            "gold": row["gold"],
            "prompt": prompt_for(statement, chunk_parts(docs, None)),
        })
        if not bool(row["evidence_in_topk"]):
            continue
        jobs.append({
            "condition": "evidence_removed",
            "claim_id": claim_id,
            "gold": row["gold"],
            "prompt": prompt_for(statement, chunk_parts(docs, evidence)),
        })

    subset_ids = set(stayed_supported(pd.read_csv(ROOT / "results" / "intervention.csv"))["claim_id"])
    doi_by_id = predictions.set_index("claim_id")["doi"]
    for _, row in removals.iterrows():
        claim_id = row["claim_id"]
        if claim_id not in subset_ids:
            continue
        statement = claims.loc[claim_id, "statement"]
        evidence = claims.loc[claim_id, "evidence"]
        docs = docs_for(claim_id, doi_by_id.loc[claim_id])
        chosen = select_sentences(
            shap_frame[shap_frame["claim_id"] == claim_id],
            int(row["n_requested"]),
            row["selection"],
            claim_id,
        )
        drop = set(zip(chosen["passage"].tolist(), chosen["sentence"].tolist()))
        jobs.append({
            "condition": f"{row['selection']}_{int(row['n_requested'])}",
            "claim_id": claim_id,
            "gold": row["gold"],
            "prompt": prompt_for(statement, sentence_parts(docs, evidence, drop)),
        })
    return jobs


def draw(llm: ChatOpenAI, prompt: str) -> tuple[str, float]:
    system = SystemMessage(content=SYSTEM)
    last_error = None
    for attempt in range(6):
        try:
            message = llm.invoke([system, HumanMessage(content=prompt)])
            return label_and_confidence(message)
        except Exception as error:
            last_error = error
            time.sleep(min(60, 2**attempt))
    raise RuntimeError(last_error)


def run_sampling() -> pd.DataFrame:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        top_p=0.9,
        logprobs=True,
        max_tokens=24,
    )
    done = load_done()
    jobs = build_jobs()
    print(f"{len(jobs)} prompts", flush=True)
    for job in jobs:
        key = (job["condition"], job["claim_id"])
        have = done.get(key, [])
        for sample_index in range(len(have), N_SAMPLES):
            label, confidence = draw(llm, job["prompt"])
            row = {
                "condition": job["condition"],
                "claim_id": job["claim_id"],
                "gold": job["gold"],
                "sample": sample_index,
                "label": label,
                "confidence": confidence,
            }
            append_sample(row)
            have.append(row)
        done[key] = have
        majority, agreement, confidence = aggregate(have)
        print(
            f"{job['condition']} {job['claim_id']} {majority} agreement={agreement:.2f} confidence={confidence:.3f}",
            flush=True,
        )
        if sum(len(rows) >= N_SAMPLES for rows in done.values()) % 20 == 0:
            write_tables(done)
    return write_tables(done)


def write_tables(done: dict[tuple[str, str], list[dict]]) -> pd.DataFrame:
    samples = [row for rows in done.values() for row in rows]
    sample_frame = pd.DataFrame(samples)
    sample_frame.to_csv(SAMPLES_PATH, index=False)
    summary_rows = []
    for (condition, claim_id), rows in done.items():
        if len(rows) < N_SAMPLES:
            continue
        majority, agreement, confidence = aggregate(rows[:N_SAMPLES])
        gold = rows[0]["gold"]
        summary_rows.append({
            "condition": condition,
            "claim_id": claim_id,
            "gold": gold,
            "majority": majority,
            "agreement": agreement,
            "confidence": confidence,
            "gold_match": majority == gold,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_PATH, index=False)
    order = ["original", "evidence_removed"]
    for selection in ("highest", "lowest", "random"):
        for n in (1, 3, 5, 7, 10):
            order.append(f"{selection}_{n}")
    table_rows = []
    for condition in order:
        group = summary[summary["condition"] == condition]
        if group.empty:
            continue
        table_rows.append({
            "condition": condition,
            "claims": int(len(group)),
            "supported": int((group["majority"] == "supported").sum()),
            "not_supported": int((group["majority"] == "not supported").sum()),
            "unclear": int((group["majority"] == "unclear").sum()),
            "gold_match": int(group["gold_match"].sum()),
            "agreement_mean": round(float(group["agreement"].mean()), 3),
            "confidence_mean": round(float(group["confidence"].mean()), 3),
        })
    table = pd.DataFrame(table_rows)
    table.to_csv(RESULTS_PATH, index=False)
    return table


if __name__ == "__main__":
    table = run_sampling()
    print(table.to_string(index=False), flush=True)
