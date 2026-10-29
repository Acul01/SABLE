"""Confirm CC BY on the publisher page for every selected paper."""

from __future__ import annotations

import json
import re
import ssl
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTED = ROOT / "data/selection/selected.json"
OUT = ROOT / "data/selection/license_checks.jsonl"

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
LICENSE_URL = re.compile(
    r"(?:https?:)?//creativecommons\.org/licenses/([a-z0-9.\-]+)",
    re.I,
)
CC_BY_PHRASE = re.compile(
    r"creative commons attribution(?!\s*[-–]?\s*non|\s*[-–]?\s*noderiv|\s*[-–]?\s*share)"
    r"|cc\s*by\s*4\.0|cc\s*by\s*3\.0|cc-by-4\.0|cc-by-3\.0",
    re.I,
)
RESTRICTIVE = re.compile(
    r"cc\s*by\s*[-–]\s*nc|cc-by-nc|noncommercial|non-commercial|"
    r"noderivatives|no derivatives|sharealike|share-alike",
    re.I,
)


def family(code: str) -> str:
    code = code.lower().strip("/")
    return code.split("/")[0]


def classify(html: str) -> dict:
    codes = []
    for match in LICENSE_URL.finditer(html):
        codes.append(family(match.group(1)))
    # unique, keep order
    seen = []
    for code in codes:
        if code not in seen:
            seen.append(code)
    exact = [c for c in seen if c == "by"]
    other = [c for c in seen if c != "by"]
    phrase = bool(CC_BY_PHRASE.search(html))
    restrictive_phrase = bool(RESTRICTIVE.search(html))
    if exact and not other:
        verdict = "cc-by"
    elif exact and other:
        verdict = "mixed"
    elif other and not exact:
        verdict = "not-cc-by"
    elif phrase and not restrictive_phrase:
        verdict = "cc-by-phrase"
    elif restrictive_phrase and not phrase:
        verdict = "restrictive-phrase"
    else:
        verdict = "not-found"
    evidence = ""
    m = LICENSE_URL.search(html)
    if m:
        evidence = m.group(0)[:240]
    elif phrase:
        evidence = CC_BY_PHRASE.search(html).group(0)[:240]
    return {
        "verdict": verdict,
        "license_codes": seen,
        "evidence": evidence,
    }


def fetch(url: str) -> tuple[str, str, int]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": UA, "Accept": "text/html,application/xhtml+xml"},
    )
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=40, context=ctx) as resp:
        final = resp.geturl()
        status = getattr(resp, "status", 200)
        raw = resp.read(1_500_000)
    html = raw.decode("utf-8", "replace")
    return final, html, status


def check_one(row: dict) -> dict:
    doi = row["doi"]
    url = f"https://doi.org/{doi}"
    result = {
        "doi": doi,
        "requested": url,
        "final_url": "",
        "status": None,
        "error": "",
        "verdict": "not-found",
        "license_codes": [],
        "evidence": "",
    }
    try:
        final, html, status = fetch(url)
        result["final_url"] = final
        result["status"] = status
        result.update(classify(html))
    except Exception as exc:  # network or HTTP error
        result["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return result


def main() -> None:
    rows = json.loads(SELECTED.read_text())
    done = {}
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                obj = json.loads(line)
                done[obj["doi"]] = obj
    pending = [r for r in rows if r["doi"] not in done or done[r["doi"]].get("error")]
    print(f"pending {len(pending)} of {len(rows)}", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(check_one, row): row["doi"] for row in pending}
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            print(result["doi"], result["verdict"], result["error"][:80], flush=True)
    # rewrite file with latest
    by_doi = dict(done)
    for result in results:
        by_doi[result["doi"]] = result
    with OUT.open("w") as fh:
        for row in rows:
            if row["doi"] in by_doi:
                fh.write(json.dumps(by_doi[row["doi"]], ensure_ascii=False) + "\n")
    from collections import Counter
    print(Counter(by_doi[r["doi"]]["verdict"] for r in rows))


if __name__ == "__main__":
    main()
