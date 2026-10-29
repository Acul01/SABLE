"""Sentence-level SHAP for claims whose label stayed supported after evidence removal.

The value is the log-probability of the label token "supported". Players are
sentences in the five retrieved chunks, with the evidence span already deleted.
"""

from __future__ import annotations

import json
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "data" / "shap_cache.jsonl"
SHAP_PATH = ROOT / "results" / "shap_sentences.csv"
INTERVENTION2_PATH = ROOT / "results" / "intervention_2.csv"
REMOVAL_PATH = ROOT / "results" / "shap_removal.csv"
RESULTS_PATH = ROOT / "results" / "results.csv"

SYSTEM = """You check one claim against five passages from the same paper.
Reply with JSON only, no markdown:
{"label": "supported" | "not supported" | "unclear", "rationale": "<one or two sentences>"}

supported: a passage reports the paper's own result and that result matches the claim.
not supported: the paper tested the claim and a passage reports the opposite or no effect.
unclear: the passages bear on the claim but do not settle it.
Use only these passages."""

FLOOR_GAP = 5.0


def split_sentences(text: str) -> list[str]:
    """Split on sentence ends. Skip abbreviations, decimals, and short fragments."""
    spans: list[str] = []
    start = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char in ".!?":
            if char == "." and _is_abbrev(text, index):
                index += 1
                continue
            piece = text[start : index + 1].strip()
            if piece:
                spans.append(piece)
            start = index + 1
            while start < len(text) and text[start].isspace():
                start += 1
            index = start
            continue
        index += 1
    tail = text[start:].strip()
    if tail:
        spans.append(tail)
    return _merge_short(spans)


def _is_abbrev(text: str, period: int) -> bool:
    window = text[max(0, period - 8) : period].lower()
    if window.endswith(("fig", "et al", "i.e", "e.g")):
        return True
    after = text[period + 1] if period + 1 < len(text) else ""
    before = text[period - 1] if period > 0 else ""
    return after.isdigit() and (before.isdigit() or before.isspace() or before in "−-=")


def _merge_short(spans: list[str]) -> list[str]:
    merged: list[str] = []
    for span in spans:
        if len(span) < 40 and merged:
            merged[-1] = f"{merged[-1]} {span}"
        else:
            merged.append(span)
    if len(merged) >= 2 and len(merged[0]) < 40:
        merged[1] = f"{merged[0]} {merged[1]}"
        merged = merged[1:]
    return [span for span in merged if span.strip()]


def passages_without_evidence(docs, evidence: str) -> list[list[str]]:
    passages = []
    removed = 0
    for doc in docs:
        text = doc.page_content
        if evidence in text:
            text = text.replace(evidence, " ")
            removed += 1
        sentences = split_sentences(text) or ["(no text)"]
        passages.append(sentences)
    if removed == 0:
        raise RuntimeError("evidence span was not in the retrieved chunks")
    return passages


def render(passages: list[list[str]], mask: np.ndarray) -> str:
    parts = []
    cursor = 0
    for number, sentences in enumerate(passages, start=1):
        kept = []
        for sentence in sentences:
            if mask[cursor] >= 0.5:
                kept.append(sentence)
            cursor += 1
        body = " ".join(kept) if kept else "(no text)"
        parts.append(f"Passage {number}:\n{body}")
    return "\n\n".join(parts)


def load_cache() -> dict[tuple[str, str], float]:
    cache: dict[tuple[str, str], float] = {}
    if not CACHE_PATH.exists():
        return cache
    for line in CACHE_PATH.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cache[(row["claim_id"], row["mask"])] = row["value"]
    return cache


def append_cache(claim_id: str, mask: str, value: float) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("a") as handle:
        handle.write(json.dumps({"claim_id": claim_id, "mask": mask, "value": value}) + "\n")


def label_logprob(message) -> float:
    """Log-probability of starting the label with 'supported'."""
    content = (message.response_metadata.get("logprobs") or {}).get("content") or []
    text = ""
    waiting = False
    floor = -20.0
    for token in content:
        text += token["token"]
        tops = token.get("top_logprobs") or []
        if tops:
            floor = min(item["logprob"] for item in tops) - FLOOR_GAP
        if not waiting and text.rstrip().endswith('{"label": "'):
            waiting = True
            continue
        if waiting:
            for alt in tops:
                if alt["token"].lstrip().startswith("supported"):
                    return float(alt["logprob"])
            if token["token"].lstrip().startswith("supported"):
                return float(token["logprob"])
            return float(floor)
    return float(floor)


class Scorer:
    def __init__(self, llm: ChatOpenAI, cache: dict[tuple[str, str], float]):
        self.llm = llm
        self.cache = cache
        self.system = SystemMessage(content=SYSTEM)

    def score(self, claim_id: str, statement: str, passages: list[list[str]], mask: np.ndarray) -> float:
        key = (claim_id, "".join("1" if bit >= 0.5 else "0" for bit in mask))
        if key in self.cache:
            return self.cache[key]
        prompt = f"Claim:\n{statement}\n\n{render(passages, mask)}"
        message = None
        for attempt in range(3):
            try:
                message = self.llm.invoke([self.system, HumanMessage(content=prompt)])
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
        value = label_logprob(message)
        self.cache[key] = value
        append_cache(claim_id, key[1], value)
        return value


def explain_claim(scorer: Scorer, claim_id: str, statement: str, passages: list[list[str]]) -> list[dict]:
    flat = [sentence for passage in passages for sentence in passage]
    count = len(flat)
    def values(masks: np.ndarray) -> np.ndarray:
        masks = np.atleast_2d(masks)
        return np.array([
            scorer.score(claim_id, statement, passages, mask) for mask in masks
        ])

    np.random.seed(zlib.adler32(claim_id.encode()) % (2**32))
    explainer = shap.KernelExplainer(values, np.zeros((1, count)))
    shap_values = explainer.shap_values(
        np.ones((1, count)),
        nsamples=max(2 * count, 40),
        l1_reg="num_features(10)",
        silent=True,
    )
    shap_values = np.array(shap_values).reshape(-1)
    rows = []
    cursor = 0
    for passage_number, sentences in enumerate(passages, start=1):
        for sentence in sentences:
            rows.append({
                "claim_id": claim_id,
                "passage": passage_number,
                "sentence_index": cursor,
                "sentence": sentence,
                "shap_value": float(shap_values[cursor]),
            })
            cursor += 1
    return rows


def stayed_supported(intervention: pd.DataFrame) -> pd.DataFrame:
    stayed = intervention[
        (intervention["flipped"] == False) & (intervention["predicted_without_evidence"] == "supported")
    ].copy()
    return stayed


def open_store() -> tuple[Chroma, int]:
    store = Chroma(
        persist_directory=str(ROOT / "chroma_db"),
        embedding_function=OpenAIEmbeddings(),
    )
    return store, 5


def run_sentence_shap() -> pd.DataFrame:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    claims = pd.read_csv(ROOT / "data" / "CLAIMS.csv").set_index("claim_id")
    subset = stayed_supported(pd.read_csv(ROOT / "results" / "intervention.csv"))
    done: set[str] = set()
    existing = []
    if SHAP_PATH.exists():
        existing_frame = pd.read_csv(SHAP_PATH)
        existing = existing_frame.to_dict("records")
        done = set(existing_frame["claim_id"])
    store, k = open_store()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, logprobs=True, top_logprobs=5, max_tokens=24)
    scorer = Scorer(llm, load_cache())
    rows = list(existing)
    for _, row in subset.iterrows():
        claim_id = row["claim_id"]
        if claim_id in done:
            continue
        statement = claims.loc[claim_id, "statement"]
        evidence = claims.loc[claim_id, "evidence"]
        docs = store.similarity_search(statement, k=k, filter={"source": row["doi"]})
        passages = passages_without_evidence(docs, evidence)
        explained = explain_claim(scorer, claim_id, statement, passages)
        rows.extend(explained)
        pd.DataFrame(rows).to_csv(SHAP_PATH, index=False)
        top = sorted(explained, key=lambda item: item["shap_value"], reverse=True)[:3]
        print(f"{claim_id} sentences={len(explained)} top={top[0]['shap_value']:.3f}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(SHAP_PATH, index=False)
    return frame


def top_positive(sentences: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    positive = sentences[sentences["shap_value"] > 0].sort_values("shap_value", ascending=False)
    return positive.head(n)


def select_sentences(sentences: pd.DataFrame, n: int, selection: str, claim_id: str) -> pd.DataFrame:
    """Highest uses only positive values. Lowest and random are nested in n."""
    if selection == "highest":
        return top_positive(sentences, n)
    seed = zlib.adler32(f"{claim_id}:{selection}".encode())
    frame = sentences.copy()
    frame["tie"] = np.random.default_rng(seed).random(len(frame))
    if selection == "lowest":
        frame = frame.sort_values(["shap_value", "tie"], ascending=[True, True])
    elif selection == "random":
        frame = frame.sort_values("tie")
    else:
        raise ValueError(selection)
    return frame.head(n)


def run_intervention_2(n: int = 3, selection: str = "highest") -> pd.DataFrame:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    claims = pd.read_csv(ROOT / "data" / "CLAIMS.csv").set_index("claim_id")
    subset = stayed_supported(pd.read_csv(ROOT / "results" / "intervention.csv"))
    shap_frame = pd.read_csv(SHAP_PATH)
    store, k = open_store()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    system = SystemMessage(content=SYSTEM)
    rows = []
    for _, row in subset.iterrows():
        claim_id = row["claim_id"]
        statement = claims.loc[claim_id, "statement"]
        evidence = claims.loc[claim_id, "evidence"]
        docs = store.similarity_search(statement, k=k, filter={"source": row["doi"]})
        passages = passages_without_evidence(docs, evidence)
        flat = [(passage_number, sentence) for passage_number, sentences in enumerate(passages, start=1) for sentence in sentences]
        chosen = select_sentences(shap_frame[shap_frame["claim_id"] == claim_id], n, selection, claim_id)
        drop = set(zip(chosen["passage"].tolist(), chosen["sentence"].tolist()))
        kept_passages: dict[int, list[str]] = {number: [] for number in range(1, 6)}
        removed = []
        for passage_number, sentence in flat:
            if (passage_number, sentence) in drop:
                removed.append(sentence)
                continue
            kept_passages[passage_number].append(sentence)
        parts = []
        for number in range(1, 6):
            body = " ".join(kept_passages[number]) if kept_passages[number] else "(no text)"
            parts.append(f"Passage {number}:\n{body}")
        reply = None
        for attempt in range(3):
            try:
                reply = llm.invoke([
                    system,
                    HumanMessage(content=f"Claim:\n{statement}\n\n" + "\n\n".join(parts)),
                ])
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
        text = reply.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(text)
        flipped = parsed["label"] != row["predicted_without_evidence"]
        rows.append({
            "claim_id": claim_id,
            "gold": row["gold"],
            "selection": selection,
            "n_requested": n,
            "n_removed": len(removed),
            "predicted_without_evidence": row["predicted_without_evidence"],
            "predicted": parsed["label"],
            "flipped": flipped,
            "removed_sentences": " || ".join(removed),
        })
        print(
            f"{selection} n={n} {claim_id} {row['predicted_without_evidence']} -> {parsed['label']}"
            f"{'  flipped' if flipped else ''}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    if selection == "highest" and n == 3:
        legacy = frame.rename(columns={"predicted": "predicted_without_top_shap"})
        legacy = legacy.drop(columns=["selection", "n_requested", "n_removed"])
        legacy.to_csv(INTERVENTION2_PATH, index=False)
    return frame


def run_ablations(ns: tuple[int, ...] = (1, 3, 5, 7, 10)) -> pd.DataFrame:
    """Add lowest-SHAP and random removals. Highest-SHAP rows are reused."""
    existing = pd.read_csv(REMOVAL_PATH)
    if "selection" not in existing.columns:
        existing["selection"] = "highest"
    frames = [existing]
    for selection in ("lowest", "random"):
        for n in ns:
            already = existing[(existing["selection"] == selection) & (existing["n_requested"] == n)]
            if len(already) >= 32:
                print(f"reuse {selection} n={n}", flush=True)
                continue
            part = run_intervention_2(n, selection)
            frames.append(part)
            pd.concat(frames, ignore_index=True).to_csv(REMOVAL_PATH, index=False)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(["selection", "n_requested", "claim_id"], keep="last")
    combined = combined.sort_values(["selection", "n_requested", "claim_id"])
    combined.to_csv(REMOVAL_PATH, index=False)
    return combined


def run_shap_removals(ns: tuple[int, ...] = (1, 3, 5, 7, 10)) -> pd.DataFrame:
    """Label the same 32 claims after dropping the top-n positive SHAP sentences."""
    frames = []
    if 3 in ns and INTERVENTION2_PATH.exists():
        cached = pd.read_csv(INTERVENTION2_PATH)
        cached["n_requested"] = 3
        cached["n_removed"] = cached["removed_sentences"].map(lambda text: 0 if pd.isna(text) or text == "" else text.count(" || ") + 1)
        cached = cached.rename(columns={"predicted_without_top_shap": "predicted"})
        frames.append(cached)
        ns = tuple(n for n in ns if n != 3)
        print("reused intervention_2.csv for n=3", flush=True)
    for n in ns:
        frames.append(run_intervention_2(n))
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["n_requested", "claim_id"])
    combined.to_csv(REMOVAL_PATH, index=False)
    return combined


def results_table(removals: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per condition, on the 32 claims whose label stayed supported."""
    if removals is None:
        removals = pd.read_csv(REMOVAL_PATH)
    if "selection" not in removals.columns:
        removals["selection"] = "highest"
    subset = stayed_supported(pd.read_csv(ROOT / "results" / "intervention.csv"))
    predictions = pd.read_csv(ROOT / "results" / "predictions.csv").set_index("claim_id")
    ids = subset["claim_id"]
    rows = []

    def add_row(condition: str, labels: pd.Series, gold: pd.Series, removed: int | None = None) -> None:
        rows.append({
            "condition": condition,
            "claims": int(len(labels)),
            "sentences_removed_min": removed if removed is not None else "",
            "supported": int((labels == "supported").sum()),
            "not_supported": int((labels == "not supported").sum()),
            "unclear": int((labels == "unclear").sum()),
            "gold_match": int((labels.to_numpy() == gold.to_numpy()).sum()),
        })

    original = predictions.loc[ids]
    add_row("original top-5", original["predicted"], original["gold"], 0)
    add_row("evidence removed", subset.set_index("claim_id").loc[ids, "predicted_without_evidence"], subset.set_index("claim_id").loc[ids, "gold"], 0)
    labels = {"highest": "highest SHAP", "lowest": "lowest SHAP", "random": "random"}
    grouped = removals.groupby(["selection", "n_requested"], sort=False)
    for selection in ("highest", "lowest", "random"):
        for n in (1, 3, 5, 7, 10):
            if (selection, n) not in grouped.groups:
                continue
            group = grouped.get_group((selection, n)).set_index("claim_id").loc[ids]
            add_row(
                f"{labels[selection]} {int(n)} removed",
                group["predicted"],
                group["gold"],
                int(group["n_removed"].min()),
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_PATH, index=False)
    return frame


if __name__ == "__main__":
    frame = run_sentence_shap()
    print("shap rows", len(frame), "claims", frame["claim_id"].nunique(), flush=True)
    second = run_intervention_2()
    flips = int(second["flipped"].sum())
    print(f"{flips} / {len(second)} labels changed after the top SHAP sentences were removed.", flush=True)
