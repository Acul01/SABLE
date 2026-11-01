"""Checks that do not call the API."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sable.agents import checker_user, extractor_user
from sable.orchestrator import revision_feedback, run_claim
from sable.schemas import Claim, Extraction, LabelScore, Passage, Review
from sable.tools import (
    check_quotes,
    complete_json,
    delete_span,
    label_token_probability,
    parse_scored,
    require_label_first,
    retrieve_passages,
)
from scripts.sentence_shap import SYSTEM


def message(content: str, probability: float = 0.5):
    return SimpleNamespace(
        content=content,
        response_metadata={"logprobs": {"content": [
            {"token": '{"label": "', "logprob": -0.01},
            {"token": "supported", "logprob": math.log(probability)},
        ]}},
    )


class FakeModel:
    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.users: list[str] = []
        self.systems: list[str] = []

    def invoke(self, messages):
        self.systems.append(messages[0].content)
        self.users.append(messages[-1].content)
        return message(self.replies.pop(0))


class ToolTests(unittest.TestCase):
    def test_quote_hit_and_miss(self):
        passages = [Passage(index=1, text="alpha beta"), Passage(index=2, text="gamma")]
        hits = check_quotes(["beta", "nope", ""], passages)
        self.assertEqual([(hit.found, hit.passage_index) for hit in hits], [
            (True, 1),
            (False, None),
            (False, None),
        ])

    def test_delete_span_counts_passages(self):
        passages = [
            Passage(index=1, text="keep SPAN here SPAN"),
            Passage(index=2, text="untouched"),
        ]
        result = delete_span(passages, "SPAN")
        self.assertEqual(result.removed, 1)
        self.assertEqual(result.passages[0].text, "keep   here  ")
        self.assertEqual(result.passages[1].text, "untouched")
        missing = delete_span(passages, "absent")
        self.assertEqual(missing.removed, 0)
        self.assertEqual(missing.passages[1].text, "untouched")

    def test_retrieve_filters_on_doi(self):
        class Store:
            def similarity_search(self, query, k, filter):
                self.query = query
                self.k = k
                self.filter = filter
                return [SimpleNamespace(page_content="chunk text")]

        store = Store()
        found = retrieve_passages("the claim", "10.1/abc", k=5, store=store)
        self.assertEqual(store.query, "the claim")
        self.assertEqual(store.k, 5)
        self.assertEqual(store.filter, {"source": "10.1/abc"})
        self.assertEqual(found[0].index, 1)
        self.assertEqual(found[0].text, "chunk text")

    def test_label_token_probability_is_the_next_token(self):
        parsed = parse_scored(
            message('{"label": "supported", "rationale": "because"}', 0.5),
            LabelScore,
        )
        self.assertEqual(parsed.label, "supported")
        self.assertAlmostEqual(parsed.token_probability, 0.5)
        self.assertAlmostEqual(label_token_probability(message('{"label": "supported", "rationale": "because"}')), math.exp(math.log(0.5)))

    def test_label_must_be_first_key(self):
        with self.assertRaises(ValueError):
            require_label_first('{"rationale": "x", "label": "supported"}')

    def test_validation_retry_sends_the_error(self):
        model = FakeModel([
            "not json",
            '{"label": "supported", "rationale": "because"}',
        ])
        from sable.schemas import LabelScore
        parsed = complete_json(
            model,
            SYSTEM,
            "Claim:\nx",
            lambda reply: parse_scored(reply, LabelScore),
        )
        self.assertEqual(parsed.label, "supported")
        self.assertEqual(model.systems[0], SYSTEM)
        self.assertIn("Validation failed", model.users[1])

    def test_checker_prompt_has_no_passages(self):
        text = checker_user("the claim", "supported", "because", ["quoted bit"])
        self.assertNotIn("Passage 1:", text)
        self.assertIn("quoted bit", text)
        from sable.agents import CHECKER_SYSTEM
        self.assertIn("You do not see the paper.", CHECKER_SYSTEM)
        self.assertNotIn("SECRET", text)


class LoopTests(unittest.TestCase):
    def claim(self):
        return Claim(claim_id="c000", doi="10.1/abc", statement="the claim")

    def passages(self):
        return [Passage(index=1, text="alpha SECRET")]

    def test_agreement_accepts(self):
        result = run_claim(
            self.claim(),
            retrieve=lambda statement, doi: self.passages(),
            extract=lambda statement, passages, feedback=None, **_: Extraction(
                label="supported", rationale="because", quotes=["alpha"], token_probability=0.8,
            ),
            review=lambda statement, label, rationale, quotes, **_: Review(
                label="supported", rationale="yes", token_probability=0.7,
            ),
            max_rounds=2,
        )
        self.assertEqual(result.decision, "accept")
        self.assertEqual(result.label, "supported")
        self.assertEqual(result.rounds, 1)
        self.assertTrue(result.quotes_ok)

    def test_second_round_can_repair_a_quote(self):
        seen = {}

        def extract(statement, passages, feedback=None, **_):
            seen["feedback"] = feedback
            seen["passages"] = passages
            quote = "alpha" if feedback else "nope"
            return Extraction(
                label="supported", rationale="because", quotes=[quote], token_probability=0.8,
            )

        result = run_claim(
            self.claim(),
            retrieve=lambda statement, doi: self.passages(),
            extract=extract,
            review=lambda statement, label, rationale, quotes, **_: Review(
                label="supported", rationale="yes", token_probability=0.7,
            ),
        )
        self.assertEqual(result.decision, "accept")
        self.assertEqual(result.rounds, 2)
        self.assertIn("nope", seen["feedback"])
        self.assertNotIn("SECRET", seen["feedback"])
        self.assertIn("SECRET", seen["passages"][0].text)

    def test_two_disagreements_abstain(self):
        result = run_claim(
            self.claim(),
            retrieve=lambda statement, doi: self.passages(),
            extract=lambda statement, passages, feedback=None, **_: Extraction(
                label="supported", rationale="because", quotes=["alpha"], token_probability=0.4,
            ),
            review=lambda statement, label, rationale, quotes, **_: Review(
                label="unclear", rationale="thin", token_probability=0.6,
            ),
            max_rounds=2,
        )
        self.assertEqual(result.decision, "abstain")
        self.assertIsNone(result.label)
        self.assertEqual(result.extractor_label, "supported")
        self.assertEqual(result.checker_label, "unclear")
        self.assertEqual(result.rounds, 2)

    def test_one_round_cap_abstains_immediately(self):
        result = run_claim(
            self.claim(),
            retrieve=lambda statement, doi: self.passages(),
            extract=lambda statement, passages, feedback=None, **_: Extraction(
                label="supported", rationale="because", quotes=["alpha"], token_probability=0.4,
            ),
            review=lambda statement, label, rationale, quotes, **_: Review(
                label="not supported", rationale="no", token_probability=0.6,
            ),
            max_rounds=1,
        )
        self.assertEqual(result.rounds, 1)
        self.assertEqual(result.decision, "abstain")

    def test_empty_retrieval_abstains_without_a_model_call(self):
        def extract(*args, **kwargs):
            raise AssertionError("extractor should not run")

        result = run_claim(
            self.claim(),
            retrieve=lambda statement, doi: [],
            extract=extract,
            review=extract,
        )
        self.assertEqual(result.decision, "abstain")
        self.assertEqual(result.rounds, 0)
        self.assertEqual([step.tool for step in result.steps], ["retrieve", "decision"])

    def test_log_is_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            run_claim(
                self.claim(),
                retrieve=lambda statement, doi: self.passages(),
                extract=lambda statement, passages, feedback=None, **_: Extraction(
                    label="supported", rationale="because", quotes=["alpha"], token_probability=0.8,
                ),
                review=lambda statement, label, rationale, quotes, **_: Review(
                    label="supported", rationale="yes", token_probability=0.7,
                ),
                log_path=path,
            )
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(rows[-1]["tool"], "decision")
        self.assertEqual(rows[-1]["output"]["decision"], "accept")

    def test_revision_note_names_the_disagreement(self):
        note = revision_feedback(
            Extraction(label="supported", rationale="because", quotes=["alpha"], token_probability=0.5),
            Review(label="unclear", rationale="thin", token_probability=0.5),
            [],
        )
        self.assertIn("supported", note)
        self.assertIn("unclear", note)
        self.assertNotIn("Passage 1:", note)

    def test_extractor_user_keeps_the_passages(self):
        text = extractor_user("the claim", [Passage(index=1, text="alpha")], "try again")
        self.assertIn("Passage 1:\nalpha", text)
        self.assertIn("Revision note:\ntry again", text)


if __name__ == "__main__":
    unittest.main()
