"""Cosine similarity between the evidence span and the SHAP sentences.

The sets are the same sentences removed in the ablation: highest SHAP,
lowest SHAP, and random, at 1, 3, 5, 7 and 10 sentences. Embeddings are the
same OpenAI model the retriever uses.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_openai import OpenAIEmbeddings

from scripts.sentence_shap import select_sentences

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "data" / "text_embeddings.npz"
SENTENCE_PATH = ROOT / "results" / "shap_similarity.csv"
RESULTS_PATH = ROOT / "results" / "similarity_results.csv"
NS = (1, 3, 5, 7, 10)


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_cache() -> dict[str, np.ndarray]:
    if not CACHE_PATH.exists():
        return {}
    stored = np.load(CACHE_PATH)
    return {key: stored[key] for key in stored.files}


def save_cache(cache: dict[str, np.ndarray]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH, **cache)


def embed_missing(texts: list[str]) -> dict[str, np.ndarray]:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    cache = load_cache()
    missing = [text for text in dict.fromkeys(texts) if text_key(text) not in cache]
    if not missing:
        return cache
    model = OpenAIEmbeddings()
    for start in range(0, len(missing), 128):
        batch = missing[start:start + 128]
        vectors = model.embed_documents(batch)
        for text, vector in zip(batch, vectors):
            cache[text_key(text)] = np.asarray(vector, dtype=np.float32)
        print(f"embedded {min(start + 128, len(missing))} / {len(missing)}", flush=True)
    save_cache(cache)
    return cache


def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left / np.linalg.norm(left, axis=1, keepdims=True)
    right = right / np.linalg.norm(right)
    return np.sum(left * right, axis=1)


def sentence_similarities() -> pd.DataFrame:
    claims = pd.read_csv(ROOT / "data" / "CLAIMS.csv").set_index("claim_id")
    shap = pd.read_csv(ROOT / "results" / "shap_sentences.csv")
    texts = list(shap["sentence"]) + [claims.loc[claim_id, "evidence"] for claim_id in shap["claim_id"].unique()]
    cache = embed_missing(texts)
    rows = []
    for claim_id, group in shap.groupby("claim_id", sort=False):
        evidence = cache[text_key(claims.loc[claim_id, "evidence"])]
        vectors = np.vstack([cache[text_key(sentence)] for sentence in group["sentence"]])
        scores = cosine(vectors, evidence)
        part = group.copy()
        part["similarity"] = scores
        rows.append(part)
    frame = pd.concat(rows, ignore_index=True)
    frame.to_csv(SENTENCE_PATH, index=False)
    return frame


def results_table(similarities: pd.DataFrame | None = None) -> pd.DataFrame:
    if similarities is None:
        similarities = pd.read_csv(SENTENCE_PATH)
    rows = []
    per_claim = []
    for selection in ("highest", "lowest", "random"):
        for n in NS:
            claim_means = []
            claim_maxes = []
            counts = []
            for claim_id, group in similarities.groupby("claim_id", sort=True):
                chosen = select_sentences(group, n, selection, claim_id)
                claim_means.append(float(chosen["similarity"].mean()))
                claim_maxes.append(float(chosen["similarity"].max()))
                counts.append(len(chosen))
                per_claim.append({
                    "claim_id": claim_id,
                    "selection": selection,
                    "n_requested": n,
                    "n_sentences": len(chosen),
                    "similarity_mean": float(chosen["similarity"].mean()),
                    "similarity_max": float(chosen["similarity"].max()),
                })
            rows.append({
                "selection": selection,
                "n_requested": n,
                "claims": int(similarities["claim_id"].nunique()),
                "sentences_mean": round(float(np.mean(counts)), 2),
                "similarity_mean": round(float(np.mean(claim_means)), 3),
                "similarity_max": round(float(np.mean(claim_maxes)), 3),
            })
    frame = pd.DataFrame(rows)
    wide = pd.DataFrame(per_claim)
    for n in NS:
        pivot = wide[wide["n_requested"] == n].pivot(index="claim_id", columns="selection", values="similarity_mean")
        for selection, other in (("highest", "lowest"), ("highest", "random")):
            mask = (frame["selection"] == selection) & (frame["n_requested"] == n)
            frame.loc[mask, f"minus_{other}"] = round(float((pivot[selection] - pivot[other]).mean()), 3)
    frame.to_csv(RESULTS_PATH, index=False)
    wide.to_csv(ROOT / "results" / "similarity_by_claim.csv", index=False)
    return frame


def run_similarity() -> pd.DataFrame:
    return results_table(sentence_similarities())


if __name__ == "__main__":
    table = run_similarity()
    print(table.to_string(index=False), flush=True)
