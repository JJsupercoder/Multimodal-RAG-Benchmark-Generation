#!/usr/bin/env python3
"""
Compute retrieval F1 from eval results and metadata.

- EVAL_BATCH_RESULTS: jsonl of evaluation outputs. Each line should contain a custom_id or id
  and somewhere the model's selected indices (e.g., JSON with key "selected_indices" or textual "Selected indices: [1,2,3]").
- METADATA_PATH: dataset JSON with entries containing gold/positive information (either explicit gold indices or candidate list).

Output: prints summary and writes `retrieval_f1_report.json`.
"""

from pathlib import Path
import json
import re
from typing import Dict, List, Optional, Set, Any, Tuple
from collections import defaultdict
import math

# ---------------- CONFIG ----------------
METADATA_PATH = Path("eval_metadata_original_webqa_4o.json")        # your enhanced dataset / metadata
EVAL_BATCH_RESULTS = Path("eval_batch_original_webqa_4o_results.jsonl")  # evaluator results with selected indices
OUTPUT_REPORT = Path("retrieval_f1_report.json")
# ----------------------------------------

# ---------------- Helpers ----------------
def read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # try to salvage if line is something like "b'...'"
                try:
                    yield json.loads(line.encode("utf-8").decode("unicode_escape"))
                except Exception:
                    print(f"Warning: failed to parse JSON on line {i} in {path}")
                    continue

def normalize_cid(raw):
    """Normalize common id shapes (strip known prefixes)."""
    if raw is None:
        return None
    if not isinstance(raw, str):
        return str(raw)
    for pfx in ("batch_req_", "id:", "guid:", "custom_id:"):
        if raw.startswith(pfx):
            return raw[len(pfx):]
    return raw

def extract_indices_from_obj(obj: Any) -> Optional[Set[int]]:
    """
    Try to extract a set of indices from a parsed JSON object `obj`.
    Looks for keys like 'selected_indices', 'selected', 'predicted', 'choices', 'selection'.
    If a value is a list of ints, use it. If a string contains numeric list, parse it.
    """
    if obj is None:
        return None
    # direct fields
    keys_to_try = [
        "selected_indices", "selected_indexes", "selected", "predicted_indices", "predicted",
        "selection", "choice_indices", "choices_indices", "choice", "selected_index",
    ]
    if isinstance(obj, dict):
        for k in keys_to_try:
            if k in obj:
                val = obj[k]
                s = _parse_indices_like(val)
                if s:
                    return s
        # sometimes nested under "result" or "output"
        for k in ("result", "data", "output", "body", "response"):
            if k in obj and isinstance(obj[k], dict):
                res = extract_indices_from_obj(obj[k])
                if res:
                    return res
    return None

def _parse_indices_like(val: Any) -> Optional[Set[int]]:
    """
    Parse a candidate value into a set of ints.
    Accepts:
      - list of ints
      - list of strings convertible to ints
      - string like "[1,2,3]" or "1,2,3" or "Selected: 1 2 3" or "selected indices: [1,3]"
    Returns None if cannot parse.
    """
    if val is None:
        return None
    # list-like
    if isinstance(val, list):
        ints = []
        for v in val:
            try:
                ints.append(int(v))
            except Exception:
                # try to extract int from string
                m = re.search(r"-?\d+", str(v))
                if m:
                    ints.append(int(m.group(0)))
        if ints:
            return set(ints)
        return None
    # numeric single
    if isinstance(val, int):
        return {val}
    # string
    s = str(val).strip()
    if not s:
        return None

    # try JSON array
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return _parse_indices_like(parsed)
    except Exception:
        pass

    # find first bracketed list like [1,2,3]
    m = re.search(r'\[ *(-?\d+(?:\s*,\s*-?\d+)*) *\]', s)
    if m:
        nums = [int(x.strip()) for x in m.group(1).split(",")]
        return set(nums)

    # find contiguous numbers separated by commas or spaces after keywords
    m2 = re.search(r'(?:selected|selection|indices|choices|chosen|picked)[^0-9\-\[]*([0-9,\s]+)', s, flags=re.I)
    if m2:
        nums = re.findall(r'-?\d+', m2.group(1))
        if nums:
            return set(int(x) for x in nums)

    # fallback: any standalone numbers in string
    nums_all = re.findall(r'-?\d+', s)
    if nums_all:
        return set(int(x) for x in nums_all)

    return None

def extract_selected_indices_from_result(res_obj: Dict[str, Any]) -> Optional[Set[int]]:
    """
    Robustly extract predicted (selected) indices from a single result object.
    Tries multiple locations/forms:
     - result['selected_indices'] or other keys
     - result['response']['body']['choices'][0]['message']['content'] parse JSON or raw text
     - result['output'] or top-level 'output' keys
    """
    # 1) direct fields in top-level
    cand = extract_indices_from_obj(res_obj)
    if cand:
        return cand

    # 2) common nested shapes
    # OpenAI-style: response.body.choices[*].message.content
    def try_content_variant(content):
        if not content:
            return None
        # if content is dict-like
        if isinstance(content, dict):
            return extract_indices_from_obj(content)
        if isinstance(content, list):
            for c in content:
                res = try_content_variant(c)
                if res:
                    return res
        if isinstance(content, str):
            # try to parse JSON blob inside string
            # e.g., content = '{"selected_indices":[1,3],"answer":"..."}'
            st = content.strip()
            # strip leading and trailing quotes
            if (st.startswith('"') and st.endswith('"')) or (st.startswith("'") and st.endswith("'")):
                st = st[1:-1]
            # try json load
            try:
                parsed = json.loads(st)
                res = extract_indices_from_obj(parsed)
                if res:
                    return res
            except Exception:
                pass
            # try regex-based parse for number lists
            return _parse_indices_like(st)
        return None

    # look for response -> body -> choices -> message -> content
    resp = res_obj.get("response") or {}
    if isinstance(resp, dict):
        body = resp.get("body") or resp
        if isinstance(body, dict):
            choices = body.get("choices")
            if isinstance(choices, list) and choices:
                for ch in choices:
                    # chat style
                    if isinstance(ch, dict):
                        # message.content
                        msg = ch.get("message") or ch.get("content") or ch.get("text")
                        if msg:
                            res = try_content_variant(msg if not isinstance(msg, dict) else msg.get("content", msg))
                            if res:
                                return res
                    else:
                        # try raw
                        res = try_content_variant(ch)
                        if res:
                            return res
            # try body.output or body.output_text
            for fallback in ("output", "output_text", "text"):
                if fallback in body:
                    res = try_content_variant(body[fallback])
                    if res:
                        return res

    # fallback top-level choices
    if "choices" in res_obj and isinstance(res_obj["choices"], list):
        for ch in res_obj["choices"]:
            # completion style
            if isinstance(ch, dict):
                # 'text' or message content
                if "text" in ch:
                    res = _parse_indices_like(ch["text"])
                    if res:
                        return res
                if "message" in ch and isinstance(ch["message"], dict):
                    res = try_content_variant(ch["message"].get("content"))
                    if res:
                        return res

    # final fallback: top-level keys 'output_text', 'output', 'text'
    for k in ("output_text", "output", "text", "content"):
        if k in res_obj:
            res = try_content_variant(res_obj[k])
            if res:
                return res

    return None

def get_gold_indices_from_metadata_entry(entry: Dict[str, Any]) -> Optional[Set[int]]:
    """
    Try to find the gold positive indices from a metadata entry.
    Heuristics:
      1) explicit keys: 'gold_indices', 'pos_indices', 'positive_indices', 'positive_idxs'
      2) 'candidates' / 'candidate_pool' list + match positives by identity (imgUrl or fact text)
      3) if dataset contains fields 'img_posFacts' and 'txt_posFacts' and also 'candidates', map by matching
    Returns set of indices (0-based) or None if cannot find.
    """
    # 1) explicit keys
    explicit_keys = ["gold_indices", "pos_indices", "positive_indices", "positive_idxs", "positive_index", "gold_idx", "gold_idx_list"]
    for k in explicit_keys:
        if k in entry:
            val = entry[k]
            parsed = _parse_indices_like(val)
            if parsed is not None:
                # ensure ints are zero-based: we won't change here; caller may adjust by candidate length heuristics
                return set(parsed)

    # 2) candidate list mapping
    candidate_keys = ["candidates", "candidate_pool", "sources", "user_content", "candidate_list"]
    candidates = None
    for ck in candidate_keys:
        if ck in entry:
            candidates = entry[ck]
            break

    # normalize candidate list to list of simple items for matching
    def candidate_to_signature(cand_item):
        """Return a tuple signature for candidate, for equality matching with positives."""
        if cand_item is None:
            return None
        if isinstance(cand_item, str):
            return ("text", cand_item.strip())
        if isinstance(cand_item, dict):
            # image url
            if cand_item.get("imgUrl"):
                return ("img", cand_item.get("imgUrl").strip())
            if cand_item.get("image_url") and isinstance(cand_item["image_url"], dict):
                return ("img", cand_item["image_url"].get("url") or cand_item["image_url"].get("imgUrl"))
            # text
            if cand_item.get("text"):
                return ("text", cand_item["text"].strip())
            # fallback to whole-dict string
            return ("dict", json.dumps(cand_item, sort_keys=True))
        # fallback
        return ("other", str(cand_item))

    if candidates:
        try:
            cand_sigs = [candidate_to_signature(c) for c in candidates]
            gold_sigs = set()
            # build gold signatures from img_posFacts and txt_posFacts
            for key in ("img_posFacts", "txt_posFacts", "posFacts", "positives"):
                if key in entry and isinstance(entry[key], list):
                    for it in entry[key]:
                        if isinstance(it, dict):
                            if it.get("imgUrl"):
                                gold_sigs.add(("img", it.get("imgUrl").strip()))
                            elif it.get("img_url"):
                                gold_sigs.add(("img", it.get("img_url").strip()))
                            elif it.get("fact"):
                                gold_sigs.add(("text", it.get("fact").strip()))
                            elif it.get("caption"):
                                gold_sigs.add(("text", it.get("caption").strip()))
                        elif isinstance(it, str):
                            gold_sigs.add(("text", it.strip()))
            # search candidate signatures for matches
            found_indices = set(i for i, s in enumerate(cand_sigs) if s in gold_sigs)
            if found_indices:
                return found_indices
        except Exception:
            pass

    # 3) Another heuristic: if entry contains 'positive_positions' or 'positive_order' keys
    alt_keys = ["positive_positions", "positive_pos", "pos_positions"]
    for k in alt_keys:
        if k in entry:
            parsed = _parse_indices_like(entry[k])
            if parsed:
                return set(parsed)

    return None

# ---------------- Metrics ----------------
def precision_recall_f1(pred: Set[int], gold: Set[int]) -> Tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0  # both empty -> perfect (or choose 0/0 semantics as you prefer)
    if not pred:
        return 0.0, 0.0, 0.0
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = (2*p*r/(p+r)) if (p+r) > 0 else 0.0
    return p, r, f1

# ---------------- Main ----------------
def main():
    if not METADATA_PATH.exists():
        print(f"Metadata file not found: {METADATA_PATH}")
        return
    if not EVAL_BATCH_RESULTS.exists():
        print(f"Eval results file not found: {EVAL_BATCH_RESULTS}")
        return

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # build mapping guid -> metadata entry
    # metadata may be dict keyed by guid, or list
    if isinstance(metadata, dict):
        meta_by_guid = metadata
    elif isinstance(metadata, list):
        meta_by_guid = {entry.get("webqa_id") or entry.get("id") or str(i): entry for i, entry in enumerate(metadata)}
    else:
        raise RuntimeError("Unsupported metadata format (expect dict or list)")

    # parse eval results
    results_by_guid = {}
    for item in read_jsonl(EVAL_BATCH_RESULTS):
        # find custom id
        cid = item.get("custom_id") or item.get("customId") or item.get("id") or None
        cid = normalize_cid(cid)
        if cid is None:
            # try nested metadata
            if isinstance(item.get("metadata"), dict):
                cid = normalize_cid(item["metadata"].get("custom_id") or item["metadata"].get("customId"))
        if cid is None:
            # skip
            continue
        # store raw item; may contain multiple choices, etc.
        results_by_guid[cid] = item

    per_guid_report = {}
    n_found = 0
    n_gold_missing = 0
    n_pred_missing = 0

    precisions = []
    recalls = []
    f1s = []

    for guid, meta_entry in meta_by_guid.items():
        # try direct match: results_by_guid uses same guid
        res_item = results_by_guid.get(guid)
        if res_item is None:
            # try using webqa_id field
            alt = meta_entry.get("webqa_id") or meta_entry.get("Guid") or meta_entry.get("id")
            if alt:
                alt = normalize_cid(alt)
                res_item = results_by_guid.get(alt)
        if res_item is None:
            # nothing predicted for this guid
            per_guid_report[guid] = {"status": "no_result"}
            n_pred_missing += 1
            continue

        # extract predicted indices
        pred_set = extract_selected_indices_from_result(res_item)
        if pred_set is None:
            # record raw text attempt: maybe content contains indices inside text
            # try to extract from text content
            content_text = None
            # try common fields
            try:
                content_text = res_item.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content")
            except Exception:
                pass
            if not content_text:
                content_text = res_item.get("response") or res_item.get("output") or res_item.get("content")
            if isinstance(content_text, str):
                pred_set = _parse_indices_like(content_text)
        if pred_set is None:
            per_guid_report[guid] = {"status": "no_predicted_indices"}
            n_pred_missing += 1
            continue

        # Normalize predicted indices: remove duplicates, convert to int, ensure non-negative
        pred_set = set(int(x) for x in pred_set if isinstance(x, (int, float)) or (isinstance(x, str) and re.match(r'^\-?\d+$', x)))
        # drop negatives
        pred_set = set(x for x in pred_set if x >= 0)

        # find gold indices
        gold_set = get_gold_indices_from_metadata_entry(meta_entry)

        # If gold_set is None try to infer via candidate list matching
        if gold_set is None:
            # if meta_entry contains 'candidates' we already handled; else try to build candidate list fallback:
            # Many metadata formats keep positives in img_posFacts/txt_posFacts and also contain 'candidates' in a sibling file.
            # We'll attempt a basic heuristic: if the metadata has 'candidate_pool' as a list of strings, attempt to match
            # candidate entries using substrings from 'img_posFacts' or 'txt_posFacts'.
            # For now, mark as missing if not found.
            per_guid_report[guid] = {
                "status": "no_gold_indices",
                "predicted": sorted(pred_set)
            }
            n_gold_missing += 1
            continue

        # At this point, both pred_set and gold_set exist; however, we must ensure indexing base alignment:
        # If predicted index values look 1-based (i.e., any index > max_index_possible), try to convert.
        # We try to determine candidate length from metadata if present:
        cand_len = None
        for ck in ("candidates", "candidate_pool", "sources", "user_content"):
            if ck in meta_entry and isinstance(meta_entry[ck], list):
                cand_len = len(meta_entry[ck])
                break
        # fallback to total known fields
        if cand_len is None:
            # approximate: sum of image/text positives+negatives if present
            cand_len = 0
            for k in ("img_posFacts", "txt_posFacts", "img_negFacts", "txt_negFacts"):
                if k in meta_entry and isinstance(meta_entry[k], list):
                    cand_len += len(meta_entry[k])

        # If predicted indices contain values > cand_len-1 and cand_len>0, assume 1-based and shift by -1
        if cand_len and any(i >= cand_len for i in pred_set):
            shifted = set(i-1 for i in pred_set)
            # if shifting made them in range, use shifted
            if not any(i >= cand_len or i < 0 for i in shifted):
                pred_set = shifted

        # similar for gold_set: if gold indices appear 1-based (>=cand_len), try shifting
        if cand_len and any(i >= cand_len for i in gold_set):
            shifted_gold = set(i-1 for i in gold_set)
            if not any(i >= cand_len or i < 0 for i in shifted_gold):
                gold_set = shifted_gold

        p, r, f1 = precision_recall_f1(pred_set, gold_set)

        per_guid_report[guid] = {
            "status": "ok",
            "predicted": sorted(pred_set),
            "gold": sorted(gold_set),
            "precision": p,
            "recall": r,
            "f1": f1,
            "n_predicted": len(pred_set),
            "n_gold": len(gold_set)
        }

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        n_found += 1

    # overall aggregates
    avg_p = float(sum(precisions)/len(precisions)) if precisions else 0.0
    avg_r = float(sum(recalls)/len(recalls)) if recalls else 0.0
    avg_f1 = float(sum(f1s)/len(f1s)) if f1s else 0.0

    summary = {
        "n_metadata": len(meta_by_guid),
        "n_results_matched": n_found,
        "n_missing_result": n_pred_missing,
        "n_missing_gold_indices": n_gold_missing,
        "avg_precision": avg_p,
        "avg_recall": avg_r,
        "avg_f1": avg_f1
    }

    report = {
        "summary": summary,
        "per_guid": per_guid_report
    }

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as outf:
        json.dump(report, outf, indent=2, ensure_ascii=False)

    print("=== Retrieval F1 Summary ===")
    for k,v in summary.items():
        print(f"{k}: {v}")
    print(f"Detailed per-guid report saved to: {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()
