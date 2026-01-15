#!/usr/bin/env python3
"""
Merge generated Q/A from results.jsonl into an existing webqa_dynamic_dataset.json.

Output: writes a new JSON file with Q (string) and A (list of strings) inserted
for each matching entry.

Usage: edit the paths below or pass them as args (simple constants used here).
"""

import json
import re
from pathlib import Path

# ==== CONFIG ====
WEBQA_FILE = Path("WebQA_dynamic_dataset_ITT_diff2.json")        # incomplete dataset (input)
RESULTS_FILE = Path("sample_batch_IIT_results.jsonl")                   # results produced by model (jsonl)
OUTPUT_FILE = Path("WebQA_dynamic_dataset_IIT_1.json")       # output merged dataset

# ==== Helpers ====

def read_jsonl(path):
    """Yield parsed JSON objects from a .jsonl file, skipping blank lines."""
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"⚠️  Could not parse JSON on line {i} of {path}: {e}")
                continue

_qa_question_re = re.compile(r'^(?:question|q)\s*[:\-]\s*(.+)$', flags=re.I)
_qa_answer_re   = re.compile(r'^(?:answer|a)\s*[:\-]\s*(.+)$', flags=re.I)

def extract_qa_from_text(text):
    """
    Try to extract Question and Answer from a block of text.
    Returns (question_or_None, answer_or_None).

    Strategy:
    - Look line-by-line for lines starting with "Question:" / "Answer:" (case-insensitive).
    - If not found, attempt to find "Question" and "Answer" markers anywhere.
    - If still not found, attempt a simple fallback: treat first non-empty line as question,
      and remaining text as answer (only if there's a clear separator like a blank line).
    - Strip quotes/extra whitespace.
    """
    if not text:
        return None, None

    # Normalize newlines and split
    lines = [ln.strip() for ln in re.split(r'\r?\n', text)]
    q, a = None, None

    # First pass: look for explicit Q/A lines
    for ln in lines:
        if not ln:
            continue
        m_q = _qa_question_re.match(ln)
        if m_q and not q:
            q = m_q.group(1).strip().strip('"').strip("'")
            continue
        m_a = _qa_answer_re.match(ln)
        if m_a and not a:
            a = m_a.group(1).strip().strip('"').strip("'")
            continue

    # If we found both, return immediately
    if q or a:
        return q, a

    # Second pass: look for "Question" / "Answer" tokens elsewhere (e.g., "Question: ... Answer: ...")
    # join into single string and try to capture with regex
    joined = " ".join([ln for ln in lines if ln])
    # pattern: Question: ... Answer: ...
    pattern = re.compile(r'(?:question\s*[:\-]\s*)(.+?)(?:\s+(?:answer|a)\s*[:\-]\s*)(.+)$', flags=re.I|re.S)
    m = pattern.search(joined)
    if m:
        q = m.group(1).strip().strip('"').strip("'")
        a = m.group(2).strip().strip('"').strip("'")
        return q, a

    # Third pass (fallback): if text contains a blank-line separator, treat part before separator as Q and after as A
    parts = re.split(r'\n\s*\n', text.strip(), maxsplit=1)
    if len(parts) == 2:
        maybe_q = parts[0].strip()
        maybe_a = parts[1].strip()
        # sanity checks: short question-ish first line
        if len(maybe_q.split()) <= 40 and len(maybe_a) > 0:
            return maybe_q, maybe_a

    # Last resort: if text is short, treat as answer only
    stripped = text.strip()
    if len(stripped.splitlines()) == 1 and len(stripped.split()) <= 40:
        # It's probably an answer only (no question). Return None for Q and the text as A
        return None, stripped

    return None, None

def extract_content_from_result(result):
    """
    Given a single parsed JSON `result` object from results.jsonl, attempt to extract
    the textual content that contains Q/A. The function tries several common shapes:
      - result["response"]["body"]["choices"][0]["message"]["content"]
      - result["response"]["choices"][0]["message"]["content"]
      - result["response"]["body"]["output"] or similar
      - result["output"] (common in other batch formats)
      - result["text"] or result["content"]
    Returns the extracted string or None.
    """
    # common nested response shapes (defensive)
    # 1) OpenAI-like: result["response"]["body"]["choices"][0]["message"]["content"]
    try:
        resp = result.get("response", {})
        if isinstance(resp, dict):
            body = resp.get("body", resp)  # sometimes response==body
            if isinstance(body, dict):
                # choices -> message -> content
                choices = body.get("choices")
                if isinstance(choices, list) and len(choices) > 0:
                    first = choices[0]
                    # chat style
                    if isinstance(first, dict) and "message" in first and isinstance(first["message"], dict):
                        content = first["message"].get("content")
                        if isinstance(content, str):
                            return content
                        # sometimes content is list/dict; join if list of text pieces
                        if isinstance(content, list):
                            return "\n".join([str(c) for c in content])
                    # completion style: 'text' or 'message' or full string
                    if isinstance(first, dict) and "text" in first and isinstance(first["text"], str):
                        return first["text"]
                    if isinstance(first, dict) and "message" in first and isinstance(first["message"], str):
                        return first["message"]
                # sometimes body has 'output' or 'output_text'
                if "output_text" in body and isinstance(body["output_text"], str):
                    return body["output_text"]
                if "output" in body:
                    out = body["output"]
                    if isinstance(out, str):
                        return out
                    if isinstance(out, list):
                        return "\n".join(str(x) for x in out)
    except Exception:
        pass

    # 2) Top-level 'choices' (older formats)
    if "choices" in result and isinstance(result["choices"], list) and result["choices"]:
        ch0 = result["choices"][0]
        # chat-style
        if isinstance(ch0, dict) and "message" in ch0 and isinstance(ch0["message"], dict):
            c = ch0["message"].get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return "\n".join([str(x) for x in c])
        # completion style
        if isinstance(ch0, dict) and "text" in ch0:
            return ch0["text"]

    # 3) other common keys
    for k in ("output_text", "output", "text", "content"):
        val = result.get(k)
        if isinstance(val, str):
            return val
        if isinstance(val, list):
            return "\n".join(str(x) for x in val)

    # 4) try to stringify some nested fields (best-effort)
    # sometimes the response is at result['response'] which is a stringified JSON
    resp = result.get("response")
    if isinstance(resp, str):
        # attempt to pull Q/A from the string
        return resp

    return None

# ==== Main merging logic ====

def main():
    if not WEBQA_FILE.exists():
        print(f"ERROR: {WEBQA_FILE} not found.")
        return
    if not RESULTS_FILE.exists():
        print(f"ERROR: {RESULTS_FILE} not found.")
        return

    # Load the incomplete dataset (should be a dict keyed by id)
    with open(WEBQA_FILE, "r", encoding="utf-8") as f:
        try:
            dataset = json.load(f)
        except Exception as e:
            print(f"ERROR: failed to parse {WEBQA_FILE}: {e}")
            return

    # Keep track of which ids were updated
    updated_ids = set()
    missing_ids = []

    # Iterate results
    for res in read_jsonl(RESULTS_FILE):
        # determine the id that maps to dataset key
        # Common fields: "custom_id", "id" (sometimes prefixed with 'batch_req_<id>')
        cid = None
        if isinstance(res, dict):
            if "custom_id" in res and res["custom_id"]:
                cid = res["custom_id"]
            elif "id" in res and res["id"]:
                raw = res["id"]
                # remove common prefixes like 'batch_req_' if present
                if isinstance(raw, str) and raw.startswith("batch_req_"):
                    cid = raw[len("batch_req_"):]
                else:
                    cid = raw
            # also allow nested metadata with 'custom_id' under top-level 'metadata' field
            elif "metadata" in res and isinstance(res["metadata"], dict) and res["metadata"].get("custom_id"):
                cid = res["metadata"].get("custom_id")

        if cid is None:
            print("⚠️  Could not determine custom id for result entry; skipping one entry.")
            continue

        # Find matching dataset entry
        if cid not in dataset:
            # Try an alternate: sometimes dataset keys are the raw GUID and custom_id contains extra
            # Try stripping a leading 'id:' or 'guid:' prefix
            alt_cid = cid
            for prefix in ("id:", "guid:"):
                if isinstance(cid, str) and cid.startswith(prefix):
                    alt_cid = cid[len(prefix):]
            if alt_cid in dataset:
                cid = alt_cid
            else:
                # record missing and continue
                missing_ids.append(cid)
                print(f"⚠️  Warning: custom_id '{cid}' not found in dataset; skipping.")
                continue

        content_text = extract_content_from_result(res)
        if content_text is None:
            # nothing to parse
            print(f"⚠️  Warning: no textual content found for result {cid}; skipping.")
            continue

        # Extract Q/A
        q_text, a_text = extract_qa_from_text(content_text)

        # If not found, try to parse JSON embedded in content (some outputs are JSON string)
        if (q_text is None and a_text is None):
            # attempt to find JSON blob inside content
            m = re.search(r'(\{.*\})', content_text, flags=re.S)
            if m:
                try:
                    obj = json.loads(m.group(1))
                    # try common keys
                    q_text = q_text or obj.get("Q") or obj.get("question") or obj.get("Question")
                    a_val = obj.get("A") or obj.get("answer") or obj.get("Answer")
                    if a_val and isinstance(a_val, list):
                        a_text = a_val[0] if a_val else None
                    elif isinstance(a_val, str):
                        a_text = a_val
                except Exception:
                    pass

        # If still no Q but have an answer-like text, place as answer only
        if q_text is None and a_text:
            pass  # keep a_text as answer
        # If Q is present but A missing, try to take subsequent lines as answer
        if q_text and not a_text:
            # try to find the Q line in the text and take what follows as answer
            pattern_q = re.compile(re.escape(q_text), flags=re.I)
            mpos = pattern_q.search(content_text)
            if mpos:
                rest = content_text[mpos.end():].strip()
                # try to pull the first Answer: line
                _, a_try = extract_qa_from_text(rest)
                if a_try:
                    a_text = a_try
                else:
                    # fallback: first non-empty line after Q
                    rest_lines = [ln.strip() for ln in re.split(r'\r?\n', rest) if ln.strip()]
                    if rest_lines:
                        a_text = rest_lines[0]

        # Final cleanup: strip and normalize
        if q_text:
            q_text = q_text.strip()
        if a_text:
            a_text = a_text.strip()

        # Insert into dataset entry
        entry = dataset[cid]
        # Set Q only if not present (or if you want to overwrite, change logic)
        entry["Q"] = q_text if q_text is not None else entry.get("Q")
        # A must be a list of strings (replace or set)
        if a_text is not None:
            entry["A"] = [a_text]
        else:
            # if no answer found, keep existing A if present, otherwise empty list
            entry["A"] = entry.get("A") or []

        updated_ids.add(cid)

    # Save merged dataset
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outf:
        json.dump(dataset, outf, indent=2, ensure_ascii=False)

    print(f"\n✅ Done. Updated {len(updated_ids)} entries. Output written to: {OUTPUT_FILE}")
    if missing_ids:
        print(f"⚠️  {len(missing_ids)} result ids were not matched to dataset keys (examples): {missing_ids[:10]}")

if __name__ == "__main__":
    main()
