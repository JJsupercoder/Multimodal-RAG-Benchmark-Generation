#!/usr/bin/env python3
"""
g_eval_summary_print.py

Read G-Eval run results (evaluation_g_eval_batch_results.jsonl), parse
the per-sample 1-5 scores returned by the model, aggregate per GUID,
and print:
  - average expectation per criterion across GUIDs
  - overall mean of those averages (single final score)

No files are written.
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Any, List
import math
from statistics import mean

# === CONFIG ===
G_EVAL_RESULTS = Path("g_eval_batch_original_webqa_4o_results.jsonl")  # input file
CRITERIA = ["correctness", "completeness", "precision", "coherence", "relevance"]
SCORE_RANGE = [1, 2, 3, 4, 5]


def parse_response_content(raw_content: str) -> Dict[str, int]:
    """Parse assistant content into a dict criterion->int (1..5)."""
    if not isinstance(raw_content, str):
        return {}
    rc = raw_content.strip()
    # Try JSON parse first
    try:
        parsed = json.loads(rc)
        if isinstance(parsed, dict):
            out = {}
            for c in CRITERIA:
                if c in parsed:
                    try:
                        v = int(parsed[c])
                        if v in SCORE_RANGE:
                            out[c] = v
                    except Exception:
                        pass
            return out
    except Exception:
        pass

    # Heuristic: key separators like "correctness": 5 or correctness = 5 or correctness:5
    out = {}
    for c in CRITERIA:
        m = re.search(rf'"?{re.escape(c)}"?\s*[:=]\s*([1-5])', rc, flags=re.IGNORECASE)
        if m:
            out[c] = int(m.group(1))

    if out:
        return out

    # Last-resort heuristics: find digits 1-5 in text
    found = re.findall(r'\b([1-5])\b', rc)
    if found:
        nums = [int(x) for x in found]
        if len(nums) == len(CRITERIA):
            return {c: v for c, v in zip(CRITERIA, nums)}
        if len(nums) == 1:
            return {c: nums[0] for c in CRITERIA}
        # choose most frequent number and assign to all (fallback)
        most = Counter(nums).most_common(1)[0][0]
        return {c: most for c in CRITERIA}

    return {}


def _get_custom_id(obj: Dict[str, Any]) -> str:
    return obj.get("custom_id") or obj.get("customId") or obj.get("id") or ""


def _base_guid_from_custom_id(cid: str) -> str:
    if not isinstance(cid, str) or not cid:
        return cid
    if "_geval_" in cid:
        return cid.split("_geval_", 1)[0]
    if cid.endswith("_geval"):
        return cid[:-len("_geval")]
    m = re.match(r"^(.*)_geval\d+$", cid)
    if m:
        return m.group(1)
    if "_geval" in cid:
        return cid.split("_geval", 1)[0]
    return cid


def load_g_eval_results(path: Path) -> Dict[str, List[Dict[str, int]]]:
    """
    Read the raw results file and build mapping:
      base_guid -> list of parsed score dicts (one per sample parsed)
    """
    mapping = defaultdict(list)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for ln_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # skip invalid JSON lines
                continue

            cid = _get_custom_id(obj)
            base = _base_guid_from_custom_id(cid)

            # Try to extract choices from typical structure
            choices = obj.get("response", {}).get("body", {}).get("choices", [])
            if not choices:
                # fallback if body.choices missing but there is content string
                content_fallback = obj.get("response", {}).get("body", {}).get("choices", [{}])
                if isinstance(content_fallback, list) and content_fallback:
                    choices = content_fallback

            # If there are choices, parse each one
            parsed_this_line = 0
            if isinstance(choices, list) and choices:
                for choice in choices:
                    # Many logs store message content at choice["message"]["content"]
                    content = ""
                    try:
                        if isinstance(choice, dict):
                            content = choice.get("message", {}).get("content", "") or choice.get("content", "")
                        else:
                            # sometimes choices are plain strings
                            content = str(choice)
                    except Exception:
                        content = ""
                    parsed = parse_response_content(content or "")
                    if parsed:
                        mapping[base].append(parsed)
                        parsed_this_line += 1

            # If nothing parsed from choices, try top-level content path
            if parsed_this_line == 0:
                try:
                    content = obj.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
                    parsed = parse_response_content(content or "")
                    if parsed:
                        mapping[base].append(parsed)
                except Exception:
                    pass

    return mapping


def aggregate_for_guid(parsed_list: List[Dict[str, int]]) -> Dict[str, Any]:
    """Aggregate parsed samples for one guid to counts/probs/expectation per criterion."""
    n = len(parsed_list)
    if n == 0:
        return {}
    counters = {c: Counter() for c in CRITERIA}
    for sample in parsed_list:
        for c in CRITERIA:
            if c in sample:
                counters[c][sample[c]] += 1
    out = {"n_samples": n, "criteria": {}}
    for c in CRITERIA:
        counts = [counters[c].get(s, 0) for s in SCORE_RANGE]
        probs = [cnt / n for cnt in counts]
        expect = sum(p * s for p, s in zip(probs, SCORE_RANGE))
        ex2 = sum((s ** 2) * p for p, s in zip(probs, SCORE_RANGE))
        var = max(0.0, ex2 - expect ** 2)
        std = math.sqrt(var)
        out["criteria"][c] = {
            "counts": {str(s): int(counters[c].get(s, 0)) for s in SCORE_RANGE},
            "probs": {str(s): float(probs[idx]) for idx, s in enumerate(SCORE_RANGE)},
            "expectation": float(expect),
            "mean": float(expect),
            "std": float(std)
        }
    return out


def compute_overall_summary(aggregated: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given aggregated per-guid results (guid -> aggregation), compute:
      - per-criterion average expectation across GUIDs
      - overall mean of these per-criterion averages
    Returns dict with per-criterion averages and overall_mean.
    """
    summary_acc = {c: {"sum_expect": 0.0, "count_guid": 0} for c in CRITERIA}
    for guid, data in aggregated.items():
        if not data:
            continue
        for c in CRITERIA:
            expect = data.get("criteria", {}).get(c, {}).get("expectation")
            if expect is not None:
                summary_acc[c]["sum_expect"] += float(expect)
                summary_acc[c]["count_guid"] += 1

    per_criterion_avg = {}
    for c in CRITERIA:
        cnt = summary_acc[c]["count_guid"]
        avg = summary_acc[c]["sum_expect"] / cnt if cnt > 0 else None
        per_criterion_avg[c] = {"avg_expectation": avg, "n_guid": cnt}

    # compute overall mean of averages (exclude None)
    valid_avgs = [v["avg_expectation"] for v in per_criterion_avg.values() if v["avg_expectation"] is not None]
    overall_mean = float(mean(valid_avgs)) if valid_avgs else None

    return {"per_criterion": per_criterion_avg, "overall_mean_of_avgs": overall_mean}


def main():
    if not G_EVAL_RESULTS.exists():
        print(f"Error: expected input file not found: {G_EVAL_RESULTS}")
        return

    mapping = load_g_eval_results(G_EVAL_RESULTS)
    if not mapping:
        print("No parsed samples found in file.")
        return

    # Aggregate per GUID
    aggregated = {}
    for guid, parsed_list in mapping.items():
        aggregated[guid] = aggregate_for_guid(parsed_list)

    # Compute overall summary
    summary = compute_overall_summary(aggregated)

    # Print results
    print("\nG-Eval aggregated summary across all GUIDs:\n")
    per_crit = summary["per_criterion"]
    for c in CRITERIA:
        info = per_crit.get(c, {})
        avg = info.get("avg_expectation")
        n_guid = info.get("n_guid", 0)
        if avg is None:
            print(f"  {c}: (no data)")
        else:
            print(f"  {c}: avg expectation = {avg:.4f}  (n_guid = {n_guid})")

    overall = summary["overall_mean_of_avgs"]
    if overall is None:
        print("\nOverall mean of per-criterion averages: (no data)")
    else:
        print(f"\nFinal overall mean (mean of the criterion averages): {overall:.4f}")

    # Also return summary dict if used programmatically
    # print("summary:", summary) 
    return summary


if __name__ == "__main__":
    main()
