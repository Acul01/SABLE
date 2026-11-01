"""Resume and summary checks. No API calls."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sable.batch import condition_summary, finished_ids


class BatchTests(unittest.TestCase):
    def test_finished_ids_keep_the_last_decision_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            rows = [
                {"claim_id": "c001", "tool": "extract", "output": {}},
                {"claim_id": "c001", "tool": "decision", "output": {"decision": "accept"}},
                {"claim_id": "c002", "tool": "error", "output": {"error": "timeout"}},
                "not json",
            ]
            path.write_text("\n".join(json.dumps(row) if isinstance(row, dict) else row for row in rows))
            self.assertEqual(finished_ids(path), {"c001"})

    def test_second_round_rescue_counts_only_new_accepts(self):
        rows_b = [
            {"claim_id": "c001", "decision": "accept", "match": "True", "extractor_label": "supported", "checker_label": "supported", "quotes_ok": "True", "rounds": "2", "v1_match": "True"},
            {"claim_id": "c002", "decision": "accept", "match": "False", "extractor_label": "supported", "checker_label": "supported", "quotes_ok": "True", "rounds": "2", "v1_match": "False"},
            {"claim_id": "c003", "decision": "abstain", "match": "", "extractor_label": "supported", "checker_label": "unclear", "quotes_ok": "False", "rounds": "2", "v1_match": "True"},
        ]
        summary = condition_summary("v2b", rows_b, rescued={"c001", "c010"})
        self.assertEqual(summary["accepted"], 2)
        self.assertEqual(summary["correct_accepts"], 1)
        self.assertEqual(summary["checker_disagreements"], 1)
        self.assertEqual(summary["quotes_failed"], 1)
        self.assertEqual(summary["rescued_by_second_round"], 1)
        self.assertEqual(summary["rescued_and_correct"], 1)


if __name__ == "__main__":
    unittest.main()
