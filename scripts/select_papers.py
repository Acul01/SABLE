"""Select CC-BY invasion-biology articles before any labeling.

Rules are documented in README.md. This script only applies them.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parents[1]
META_PATH = ROOT / "data/zenodo/ccby_ft_meta.jsonl"
CSV_PATH = ROOT / "data/zenodo/publications_tdm.csv"
OUT_DIR = ROOT / "data/selection"

INVASION_JOURNALS = {
    "Aquatic Invasions",
    "BioInvasions Records",
    "Biological Invasions",
    "NeoBiota",
    "Management of Biological Invasions",
}

TITLE_SKIP = re.compile(
    r"\b("
    r"systematic review|meta-analysis|meta analysis|editorial|commentary|"
    r"perspective|opinion|preface|foreword|correction|erratum|"
    r"letter to the editor|synthesis|conceptual framework|a framework|"
    r"papermaking|cellulose|fungicide|valuation|"
    r"introducing the|world register|likely to threaten|"
    r"options for reducing"
    r")\b|\breview\b|host invasion",
    re.I,
)
HEADER_REVIEW = re.compile(
    r"TYPE\s+Review|Article\s+Type[:\s]+Review|REVIEW ARTICLE|"
    r"REVIEW\s*published:|\bhorizon[- ]scan\b",
    re.I,
)
INVASION = re.compile(
    r"\b(invas\w*|non-native|nonnative|aliens?|introduced|"
    r"naturali[sz]ed|naturali[sz]ation)\b",
    re.I,
)
FRESH = re.compile(
    r"\b(freshwater|lakes?|rivers?|streams?|ponds?|reservoirs?|wetlands?|crayfish)\b",
    re.I,
)
PLANT = re.compile(r"\b(plants?|weeds?|flora|seedlings?|shrubs?|grasses?|trees?)\b", re.I)
MARINE = re.compile(
    r"\b(marine|ocean|oceans|coastal|seas?|estuar\w+|intertidal|reefs?|"
    r"seagrass|ballast|fouling)\b",
    re.I,
)
INSECT = re.compile(
    r"\b(insects?|ants?|beetles?|moths?|wasps?|aphids?|mosquito(?:es)?|termites?)\b",
    re.I,
)

QUOTAS = {
    "freshwater": 12,
    "plant": 12,
    "marine": 8,
    "insect": 8,
    "other": 8,
}
JOURNAL_CAP = 3
MIN_JOURNAL_PAPERS = 5
MIN_CHARS = 20_000


def theme_of(title: str) -> str:
    if FRESH.search(title) and not MARINE.search(title):
        return "freshwater"
    if PLANT.search(title):
        return "plant"
    if MARINE.search(title):
        return "marine"
    if INSECT.search(title):
        return "insect"
    return "other"


def load_meta() -> dict:
    meta = {}
    for line in META_PATH.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            meta[row["doi"]] = row
    return meta


def load_texts(dois: set[str]) -> dict[str, str]:
    raw = CSV_PATH.read_bytes().replace(b"\x00", b"").decode("utf-8", "replace")
    texts = {}
    for rec in csv.DictReader(io.StringIO(raw)):
        doi = rec["DOI"].strip()
        if doi in dois:
            texts[doi] = rec.get("Full-text") or ""
    return texts


def eligible_rows() -> list[dict]:
    meta = load_meta()
    texts = load_texts(set(meta))
    rows = []
    for doi, m in meta.items():
        if m.get("type") != "article" or m.get("language") != "en":
            continue
        if m.get("version") != "publishedVersion":
            continue
        title = m.get("title") or ""
        journal = m.get("journal") or "unknown"
        text = texts.get(doi) or ""
        if len(text.strip()) < MIN_CHARS:
            continue
        if TITLE_SKIP.search(title):
            continue
        if HEADER_REVIEW.search(text[:12_000]):
            continue
        if not (INVASION.search(title) or journal in INVASION_JOURNALS):
            continue
        rows.append(
            {
                "doi": doi,
                "title": title,
                "year": m.get("year") or 0,
                "journal": journal,
                "theme": theme_of(title),
                "landing": m.get("landing") or f"https://doi.org/{doi}",
                "pdf": m.get("pdf") or "",
                "chars": len(text.strip()),
                "openalex_license": "cc-by",
            }
        )
    counts = Counter(r["journal"] for r in rows)
    kept = [r for r in rows if counts[r["journal"]] >= MIN_JOURNAL_PAPERS]
    return kept


def select(rows: list[dict]) -> tuple[list[dict], dict[str, list[dict]]]:
    by_theme = defaultdict(list)
    for row in rows:
        by_theme[row["theme"]].append(row)
    for theme in by_theme:
        by_theme[theme].sort(key=lambda r: (-int(r["year"] or 0), r["doi"]))

    journal_count: Counter = Counter()
    selected = []
    reserves = {theme: [] for theme in QUOTAS}
    for theme, quota in QUOTAS.items():
        buckets = defaultdict(list)
        for row in by_theme[theme]:
            buckets[row["journal"]].append(row)
        picked = 0
        while picked < quota:
            candidates = [
                journal
                for journal, items in buckets.items()
                if items and journal_count[journal] < JOURNAL_CAP
            ]
            if not candidates:
                break
            candidates.sort(key=lambda journal: (journal_count[journal], journal.lower()))
            journal = candidates[0]
            selected.append(dict(buckets[journal].pop(0)))
            journal_count[journal] += 1
            picked += 1
        for journal in sorted(buckets, key=str.lower):
            reserves[theme].extend(buckets[journal])
    return selected, reserves


def main() -> None:
    rows = eligible_rows()
    selected, reserves = select(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "selected.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2)
    )
    (OUT_DIR / "reserves.json").write_text(
        json.dumps(reserves, ensure_ascii=False, indent=2)
    )
    print(f"eligible_in_populated_journals {len(rows)}")
    print(f"selected {len(selected)}")
    print(Counter(r["theme"] for r in selected))
    print(Counter(r["journal"] for r in selected).most_common())
    for i, row in enumerate(selected, 1):
        print(f"{i:02} [{row['theme']}] {row['year']} | {row['journal']} | {row['doi']}")
        print(f"    {row['title']}")


if __name__ == "__main__":
    main()
