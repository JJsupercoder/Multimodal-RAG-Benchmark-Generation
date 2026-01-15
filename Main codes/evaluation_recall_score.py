
import json
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
import numpy as np
import re
import sys
from collections import Counter

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# === CONFIG ===
ENHANCED_DATASET_PATH = Path("eval_metadata_original_webqa_4o.json")#("WebQA_enhanced_dataset.json")
EVAL_BATCH_RESULTS = Path("eval_batch_original_webqa_4o_results.jsonl")  # results from runs (contains sample answers)

# Load spaCy model (en_core_web_md as you had)
try:
    nlp = spacy.load("en_core_web_md")
except Exception as e:
    print("Failed to load en_core_web_md:", e, file=sys.stderr)
    print("Try: python -m spacy download en_core_web_md", file=sys.stderr)
    raise

# --- helpers ---------------------------------------------------------------
def load_dataset(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_eval_batch_results(path: Path) -> Dict[str, str]:
    """
    Read eval_batch_results.jsonl and extract the sample 'answer' string for each entry.
    Returns mapping: custom_id -> candidate_answer string
    """
    mapping: Dict[str, str] = {}
    if not path.exists():
        raise FileNotFoundError(f"Eval results file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for ln_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # skip invalid JSON line
                continue

            cid = obj.get("custom_id") or obj.get("customId") or obj.get("id") or ""
            candidate_answer = ""

            # try robust extraction from multiple possible structures
            # 1) response.body.choices[*].message.content (common)
            try:
                choices = obj.get("response", {}).get("body", {}).get("choices", [])
                if isinstance(choices, list) and choices:
                    # try each choice until we find an 'answer' field or plain text
                    for choice in choices:
                        content = ""
                        if isinstance(choice, dict):
                            content = choice.get("message", {}).get("content") or choice.get("content") or ""
                        else:
                            content = str(choice)
                        if not content:
                            continue
                        # if content is a JSON string like '{"selected_indices":..,"answer":"..."}'
                        parsed = None
                        if isinstance(content, str):
                            content = content.strip()
                            try:
                                parsed = json.loads(content)
                            except Exception:
                                parsed = None
                        if isinstance(parsed, dict) and "answer" in parsed:
                            candidate_answer = parsed.get("answer", "")
                            break
                        # fallback: if content contains JSON inside (escaped), attempt second parse
                        if isinstance(content, str) and content.startswith('"') and content.endswith('"'):
                            try:
                                inner = json.loads(content)  # unescape
                                if isinstance(inner, dict) and "answer" in inner:
                                    candidate_answer = inner.get("answer", "")
                                    break
                            except Exception:
                                pass
                        # last fallback: take the raw content text
                        candidate_answer = content
                        break
                else:
                    # fallback to single message content shape
                    content = obj.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
                    if content:
                        try:
                            parsed = json.loads(content)
                            candidate_answer = parsed.get("answer", "") if isinstance(parsed, dict) else content
                        except Exception:
                            candidate_answer = content
            except Exception:
                candidate_answer = ""

            if cid and candidate_answer:
                mapping[cid] = candidate_answer

    return mapping


# --- keyword extraction ---------------------------------------------------
def normalize_token(tok_text: str) -> str:
    # basic normalization: lowercase, strip, remove surrounding punctuation
    t = tok_text.strip().lower()
    t = re.sub(r"^[^\w\d]+|[^\w\d]+$", "", t)  # strip non-alnum at edges
    return t


def extract_keywords_from_text(text: str, min_token_len: int = 2) -> Set[str]:
    """
    Extract keywords as a set of normalized strings (multiword noun-chunks and named entities prioritized).
    Also include lemmatized content tokens (NOUN/PROPN/NUM/ADJ optionally).
    """
    if not text or not text.strip():
        return set()

    doc = nlp(text)
    keys = []

    # 1) Named entities (multi-word preserved)
    # for ent in doc.ents:
    #     ent_txt = normalize_token(ent.text)
    #     if ent_txt and len(ent_txt) >= min_token_len:
    #         keys.append(ent_txt)

    # # 2) Noun chunks (multi-word)
    # for nc in doc.noun_chunks:
    #     nc_txt = normalize_token(nc.text)
    #     if nc_txt and len(nc_txt) >= min_token_len:
    #         keys.append(nc_txt)

    # 3) Content word lemmas: NOUN, PROPN, NUM, ADJ (optionally include ADJ)
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        if token.pos_ in {"NOUN", "PROPN", "NUM", "ADJ", "INTJ"}:
            # use lemma where sensible
            lemma = token.lemma_.strip().lower()
            lemma = normalize_token(lemma)
            if lemma and len(lemma) >= min_token_len and lemma not in STOP_WORDS:
                keys.append(lemma)

    # deduplicate, but keep only meaningful tokens (avoid single letters)
    # set-of-strings
    kws = set(k for k in keys if k and len(k) >= min_token_len)
    return kws


# --- metrics --------------------------------------------------------------
def recall_for_sets(reference_set: Set[str], candidate_set: Set[str]) -> float:
    if not reference_set:
        return 0.0
    tp = len(reference_set & candidate_set)
    return tp / len(reference_set)


# def precision_for_sets(reference_set: Set[str], candidate_set: Set[str]) -> float:
#     if not candidate_set:
#         return 0.0
#     tp = len(reference_set & candidate_set)
#     return tp / len(candidate_set)


# def f1_from_prec_recall(p: float, r: float) -> float:
#     if p + r == 0:
#         return 0.0
#     return 2 * p * r / (p + r)


# --- main -----------------------------------------------------------------
def main():
    # print("Loading enhanced dataset...", file=sys.stderr)
    dataset = load_dataset(ENHANCED_DATASET_PATH)
    # print("Loading candidate answers from eval batch results...", file=sys.stderr)
    cand_map = load_eval_batch_results(EVAL_BATCH_RESULTS)

    recalls = []
    # precisions = []
    # f1s = []

    skipped = 0
    per_guid_stats = {}

    for guid, entry in dataset.items():
        # The dataset may use field 'A' or 'references' or 'references' list; be defensive
        references = entry.get("references") or entry.get("A") or entry.get("answers") or []
        # combine multiple references into one text string
        reference_text = " ".join(references) if isinstance(references, (list, tuple)) else str(references or "")

        # candidate answer lookup: cand_map keys are custom_id values from batch (they should match guid)
        candidate_answer = cand_map.get(guid) or cand_map.get(entry.get("webqa_id") or "") or cand_map.get(entry.get("Guid") or "")

        if not candidate_answer:
            skipped += 1
            continue

        ref_keys = extract_keywords_from_text(reference_text)
        cand_keys = extract_keywords_from_text(candidate_answer)

        if not ref_keys:
            # can't compute recall; skip but log
            skipped += 1
            per_guid_stats[guid] = {"skipped_no_reference_keywords": True}
            continue

        r = recall_for_sets(ref_keys, cand_keys)
        # p = precision_for_sets(ref_keys, cand_keys)
        # f1 = f1_from_prec_recall(p, r)

        recalls.append(r)
        # precisions.append(p)
        # f1s.append(f1)

        per_guid_stats[guid] = {
            "recall": r,
            # "precision": p,
            # "f1": f1,
            "n_ref_keys": len(ref_keys),
            "n_cand_keys": len(cand_keys)
        }

    # compute averages (ignore empty lists)
    avg_recall = float(np.mean(recalls)) if recalls else 0.0
    # avg_precision = float(np.mean(precisions)) if precisions else 0.0
    # avg_f1 = float(np.mean(f1s)) if f1s else 0.0

    print("\n=== Summary ===")
    print(f"Processed dataset entries: {len(dataset)}")
    print(f"Candidates found & evaluated: {len(recalls)}")
    print(f"Skipped entries (no candidate or no ref keywords): {skipped}")
    print(f"Average Recall : {avg_recall:.4f}")
    # print(f"Average Precision: {avg_precision:.4f}")
    # print(f"Average F1 : {avg_f1:.4f}")

    return {
        "n_dataset": len(dataset),
        "n_evaluated": len(recalls),
        "skipped": skipped,
        "avg_recall": avg_recall,
        # "avg_precision": avg_precision,
        # "avg_f1": avg_f1,
        "per_guid_stats": per_guid_stats
    }


if __name__ == "__main__":
    summary = main()
