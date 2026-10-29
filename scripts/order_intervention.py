"""Label the 67 claims again, with the rationale written before the label.

Same passages and temperature 0 as predictions.csv. Only the JSON field order changes,
so the label token is conditioned on the rationale the model has just written.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from scripts.sentence_shap import open_store

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "results" / "order_predictions.csv"
CACHE_PATH = ROOT / "data" / "order_cache.jsonl"

SYSTEM = """You check one claim against five passages from the same paper.
Reply with JSON only, no markdown. Write the rationale first, then the label:
{"rationale": "<one or two sentences>", "label": "supported" | "not supported" | "unclear"}

supported: a passage reports the paper's own result and that result matches the claim.
not supported: the paper tested the claim and a passage reports the opposite or no effect.
unclear: the passages bear on the claim but do not settle it.
Use only these passages."""


def parse_reply(raw: str) -> tuple[str, str, bool]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    parsed = json.loads(text)
    label = parsed["label"]
    rationale = parsed["rationale"]
    if label not in {"supported", "not supported", "unclear"}:
        raise ValueError(label)
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("empty rationale")
    rationale_first = text.find('"rationale"') < text.find('"label"')
    return label, rationale, rationale_first


def load_done() -> dict[str, dict]:
    done = {}
    if not CACHE_PATH.exists():
        return done
    for line in CACHE_PATH.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            done[row["claim_id"]] = row
    return done


def append_row(row: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def run_order() -> pd.DataFrame:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    claims = pd.read_csv(ROOT / "data" / "CLAIMS.csv")
    baseline = pd.read_csv(ROOT / "results" / "predictions.csv").set_index("claim_id")
    store, k = open_store()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    system = SystemMessage(content=SYSTEM)
    done = load_done()
    for _, row in claims.iterrows():
        claim_id = row["claim_id"]
        if claim_id in done:
            continue
        docs = store.similarity_search(row["statement"], k=k, filter={"source": row["doi"]})
        passages = "\n\n".join(
            f"Passage {number}:\n{doc.page_content}" for number, doc in enumerate(docs, start=1)
        )
        reply = None
        parsed = None
        for attempt in range(4):
            try:
                reply = llm.invoke([
                    system,
                    HumanMessage(content=f"Claim:\n{row['statement']}\n\n{passages}"),
                ])
                parsed = parse_reply(reply.content)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2)
        label, rationale, rationale_first = parsed
        record = {
            "claim_id": claim_id,
            "doi": row["doi"],
            "gold": row["label"],
            "predicted_label_first": baseline.loc[claim_id, "predicted"],
            "predicted_rationale_first": label,
            "changed": label != baseline.loc[claim_id, "predicted"],
            "gold_match": label == row["label"],
            "rationale_before_label": rationale_first,
            "rationale": rationale,
        }
        append_row(record)
        done[claim_id] = record
        mark = "  changed" if record["changed"] else ""
        print(f"{claim_id} {record['predicted_label_first']} -> {label}{mark}", flush=True)
    frame = pd.DataFrame(done.values()).sort_values("claim_id")
    frame.to_csv(OUT_PATH, index=False)
    changed = int(frame["changed"].sum())
    obeyed = int(frame["rationale_before_label"].sum())
    matched = int(frame["gold_match"].sum())
    print(
        f"{changed} / {len(frame)} labels changed. "
        f"{obeyed} / {len(frame)} wrote the rationale first. "
        f"{matched} / {len(frame)} match gold.",
        flush=True,
    )
    print(
        pd.crosstab(
            frame["predicted_label_first"],
            frame["predicted_rationale_first"],
            margins=True,
        ).to_string(),
        flush=True,
    )
    return frame


if __name__ == "__main__":
    run_order()
