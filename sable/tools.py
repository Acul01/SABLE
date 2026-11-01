"""The four tools. Each one wraps a function the version-1 scripts already use.

retrieve_passages wraps the DOI-filtered similarity search.
check_quotes wraps the exact substring test that defines a retrieval hit.
delete_span wraps the evidence deletion: the span becomes one space.
label_token_probability wraps the version-1 confidence, the probability of the
first token of the emitted label. It is not an entropy, and it is not the
log-probability of the token "supported".
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from scripts.sample_labels import label_and_confidence
from scripts.sentence_shap import SYSTEM, open_store
from sable.schemas import DeleteResult, LabelScore, Passage, QuoteHit

ROOT = Path(__file__).resolve().parents[1]


def ensure_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")


def passages_text(passages: list[Passage]) -> str:
    return "\n\n".join(f"Passage {passage.index}:\n{passage.text}" for passage in passages)


def retrieve_passages(statement: str, doi: str, *, k: int = 5, store=None) -> list[Passage]:
    """Top chunks from the paper with this DOI. The query is the claim."""
    if store is None:
        ensure_env()
        store, k = open_store()
    docs = store.similarity_search(statement, k=k, filter={"source": doi})
    return [
        Passage(index=number, text=doc.page_content)
        for number, doc in enumerate(docs, start=1)
    ]


def check_quotes(quotes: list[str], passages: list[Passage]) -> list[QuoteHit]:
    """A quote counts only as an exact substring of one retrieved passage."""
    hits = []
    for quote in quotes:
        passage_index = None
        if quote.strip():
            for passage in passages:
                if quote in passage.text:
                    passage_index = passage.index
                    break
        hits.append(QuoteHit(
            quote=quote,
            found=passage_index is not None,
            passage_index=passage_index,
        ))
    return hits


def delete_span(passages: list[Passage], span: str) -> DeleteResult:
    """Replace the span with one space in every passage that contains it.

    removed is the number of passages that contained the span. A missing span
    returns the passages unchanged and removed 0. The version-1 script raises
    in that case; a tool returns the count so the caller can record the miss.
    """
    edited = []
    removed = 0
    for passage in passages:
        if span and span in passage.text:
            edited.append(Passage(index=passage.index, text=passage.text.replace(span, " ")))
            removed += 1
        else:
            edited.append(passage)
    return DeleteResult(passages=edited, removed=removed)


def strip_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return text


def require_label_first(text: str) -> None:
    label_at = text.find('"label"')
    if label_at < 0:
        raise ValueError("label must be the first key")
    for key in ('"rationale"', '"quotes"'):
        other = text.find(key)
        if other >= 0 and other < label_at:
            raise ValueError("label must be the first key")


def label_token_probability(message) -> float:
    """Probability of the first token of the emitted label."""
    return label_and_confidence(message)[1]


def parse_scored(message, model: type[LabelScore]):
    text = strip_fence(message.content)
    require_label_first(text)
    data = json.loads(text)
    data.pop("token_probability", None)
    data["token_probability"] = label_token_probability(message)
    return model.model_validate(data)


def complete_json(llm, system: str, user: str, parse):
    """Call the model and parse. On a schema error, send the error back once more."""
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    last_error = None
    for attempt in range(3):
        try:
            message = llm.invoke(messages)
        except Exception as error:
            last_error = error
            if attempt == 2:
                break
            time.sleep(2)
            continue
        try:
            return parse(message)
        except (ValueError, json.JSONDecodeError, ValidationError) as error:
            last_error = error
            if attempt == 2:
                break
            messages.append(AIMessage(content=message.content))
            messages.append(HumanMessage(
                content="Validation failed: "
                f"{error}. Reply with JSON only, and put the label key first."
            ))
    raise RuntimeError(last_error)


def make_llm():
    from langchain_openai import ChatOpenAI

    ensure_env()
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        logprobs=True,
        top_logprobs=5,
    )


def score_label(statement: str, passages: list[Passage], *, llm=None) -> LabelScore:
    """The version-1 labeler: same system text, temperature 0, label before rationale."""
    model = llm if llm is not None else make_llm()
    user = f"Claim:\n{statement}\n\n{passages_text(passages)}"
    return complete_json(model, SYSTEM, user, lambda message: parse_scored(message, LabelScore))
