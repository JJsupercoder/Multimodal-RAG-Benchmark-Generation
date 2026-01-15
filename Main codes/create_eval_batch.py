# create_eval_batch.py
import json
import random
from pathlib import Path
from typing import List, Dict, Any
# from create_batch import SEED
SEED = 42

# === CONFIG ===
SAMPLE_BATCH_PATH = Path("sample_batch_IIT.jsonl")    # contains custom_id and the additional source / example Q/A
WEBQA_PATH = Path("WebQA_dynamic_dataset_IIT_1.json") # enhanced webqa (json)
OUTPUT_BATCH_PATH = Path("eval_batch_IIT.jsonl")
METADATA_PATH = Path("eval_metadata_IIT.json")
SYSTEM_PROMPT = """You are an assistant whose job is to (1) pick which of the provided candidate sources are required to answer the question, and (2) produce a concise answer using only the selected sources.  

Instructions (strict): 
- You are given a question Q and a list of candidate sources (some are images, some are text). The candidate list is anonymized: you do NOT know which sources are "positive" or "negative".
- First, select the subset of candidate indices that are required to answer Q. Indices are 0-based and correspond to the order in the candidate list. Only include indices that are necessary to answer the question.
- Second, produce a concise natural-language answer (one or two sentences maximum) that answers Q using only the selected sources. Do NOT include extra commentary.
- Output EXACTLY a JSON object with two keys: "selected_indices" (list of integers) and "answer" (string). The output must be valid JSON and nothing else. Example:
  {"selected_indices":[0,3],"answer":"Yes — both show men in 18th-century attire with wigs."}
- Do not reveal any internal reasoning, do not list sources, do not output any other text.

Expected JSON Output:
{\"selected_indices\":[...], \"answer\":\"...\" } -- nothing else.
"""

MODEL_NAME = "gpt-4.1-mini-2025-04-14" #'gpt-4o-mini-2024-07-18'
MAX_SOURCES = 12
RANDOM_SEED = SEED

# === helper loaders ===
def load_sample_batch(path: Path) -> Dict[str, Any]:
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            obj = json.loads(line)
            cid = obj.get("custom_id") or obj.get("Guid") or obj.get("id")
            if cid:
                m[cid] = obj
    return m

def load_webqa(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # If it's keyed by top-level keys with webqa_id stored in entry, create map keyed by webqa_id
    mapping = {}
    if isinstance(obj, dict):
        # some entries include 'webqa_id' inside the entry; otherwise use key as guid
        for k, v in obj.items():
            if isinstance(v, dict) and v.get("webqa_id"):
                mapping[v["webqa_id"]] = v
            else:
                # use the webqa id field or assume key is the guid
                mapping_key = v.get("webqa_id") or v.get("Guid") or k
                mapping[mapping_key] = v
    return mapping

def unify_img_item(img_entry: Dict[str, Any]) -> Dict[str, Any]:
    # create a minimal candidate representation for image
    return {
        "type": "image",
        "imgUrl": img_entry.get("imgUrl") or img_entry.get("url") or img_entry.get("image_url") or "",
        "caption": img_entry.get("caption") or img_entry.get("title") or ""
    }

def unify_txt_item(txt_entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "text",
        "text": txt_entry.get("fact") or txt_entry.get("text") or txt_entry.get("title") or ""
    }

# === Build batch entries ===
def build_candidates_for_entry(webqa_entry: Dict[str, Any], sample_entry: Dict[str, Any]) -> (List[Dict[str, Any]], List[int]):
    """
    returns (candidate_list, gold_indices)
    candidate_list: list of dicts each with type:image/text and relevant fields
    gold_indices: list of indices in candidate_list that are "gold" (posFacts + additional if positive)
    """
    candidates: List[Dict[str, Any]] = []
    gold_flags: List[int] = []

    # add positive images
    for im in webqa_entry.get("img_posFacts", []):
        candidates.append(unify_img_item(im))
        gold_flags.append(1)
    # add positive texts
    for t in webqa_entry.get("txt_posFacts", []):
        candidates.append(unify_txt_item(t))
        gold_flags.append(1)

    # add additional source from sample_entry if present
    additional = None
    # extract additional from sample_entry content
    # sample_entry has same format as before (body->messages->user->content list)
    for m in (sample_entry.get("body", {}).get("messages", []) if sample_entry else []):
        if m.get("role") == "user":
            for item in m.get("content", []):
                if isinstance(item, dict) and item.get("type", "").lower() == "text":
                    txt = item.get("text", "")
                    if txt.lower().startswith("additional text source:"):
                        additional = {"type":"text", "text": txt.split(":",1)[1].strip()}
                elif isinstance(item, dict) and item.get("type", "").lower() in ("image_url", "image"):
                    # try to detect 'additional' in following caption text
                    # treat all image_url as candidate; but mark as additional only if caption contains 'additional'
                    pass
    if additional:
        # avoid duplicates
        if all(not (c.get("type")=="text" and c.get("text")==additional.get("text")) for c in candidates):
            candidates.append(additional)
            gold_flags.append(1)

    # add negatives as distractors until max or no negs
    # interleave img_negFacts and txt_negFacts in deterministic order
    neg_imgs = webqa_entry.get("img_negFacts", []) or []
    neg_txts = webqa_entry.get("txt_negFacts", []) or []

    ni = 0
    nt = 0
    while len(candidates) < MAX_SOURCES and (ni < len(neg_imgs) or nt < len(neg_txts)):
        if ni < len(neg_imgs):
            candidates.append(unify_img_item(neg_imgs[ni]))
            gold_flags.append(0)
            ni += 1
            if len(candidates) >= MAX_SOURCES:
                break
        if nt < len(neg_txts) and len(candidates) < MAX_SOURCES:
            candidates.append(unify_txt_item(neg_txts[nt]))
            gold_flags.append(0)
            nt += 1

    # If we still have more than MAX_SOURCES (because many positives), truncate but keep positives first
    if len(candidates) > MAX_SOURCES:
        # Keep positives (gold_flags==1) in front, then keep first needed negatives.
        zipped = list(zip(candidates, gold_flags))
        positives = [z for z in zipped if z[1]==1]
        negatives = [z for z in zipped if z[1]==0]
        keep = positives + negatives[:(MAX_SOURCES - len(positives))]
        candidates, gold_flags = zip(*keep)
        candidates = list(candidates)
        gold_flags = list(gold_flags)

    # Now shuffle candidates deterministically but keep mapping to gold flags
    # rng = random.Random(RANDOM_SEED)
    combined = list(zip(candidates, gold_flags))
    random.shuffle(combined)
    candidates, gold_flags = zip(*combined)
    candidates = list(candidates)
    gold_flags = list(gold_flags)

    gold_indices = [i for i, flag in enumerate(gold_flags) if flag == 1]
    return candidates, gold_indices

def create_batch():
    random.seed(RANDOM_SEED)

    sample_map = load_sample_batch(SAMPLE_BATCH_PATH)
    webqa_map = load_webqa(WEBQA_PATH)

    batch_lines = []
    metadata = {}
    count = 0

    rand_inc = 0
    for cid, sample_entry in sample_map.items():
        # find webqa entry by custom id (some enhanced webqa use webqa_id inside)
        webqa_entry = webqa_map.get(cid)
        if webqa_entry is None:
            # try find by webqa_id inside entries
            # search mapping for entry whose 'webqa_id' equals cid
            matched = None
            for k, v in webqa_map.items():
                if isinstance(v, dict) and v.get("webqa_id") and v.get("webqa_id") == cid:
                    matched = v
                    break
            if matched:
                webqa_entry = matched
            else:
                # skip unmatched
                continue

        Q = webqa_entry.get("Q", "")
        references = webqa_entry.get("A", [])

        candidates, gold_indices = build_candidates_for_entry(webqa_entry, sample_entry)

        if not candidates:
            continue

        # build user content: instruction text + candidiate list
        user_content = []
        # instruction for the user message (short)
        user_content.append({
            "type": "text",
            "text": "You are given a question and a list of candidate sources. Pick which sources are required (indices 0-based) and answer concisely."
        })
        # Provide the question
        user_content.append({"type":"text", "text": f"Question: {Q}"})

        # add candidate list (images and texts)
        for c in candidates:
            if c.get("type") == "image":
                # include image_url item with only the url and caption as a subsequent text item
                user_content.append({"type":"image_url", "image_url": {"url": c.get("imgUrl",""), "detail": "low"}})
                # include caption if present as text
                if c.get("caption"):
                    user_content.append({"type":"text", "text": f"Caption: {c.get('caption')}"})
            else:
                # text
                user_content.append({"type":"text", "text": c.get("text","")})

        batch_entry = {
            "custom_id": cid,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL_NAME,
                "temperature": 0.0,
                "top_p": 0.0,
                # "max_completion_tokens": 200,
                "max_tokens": 200,  # adjust if you want longer answers
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
            }
        }

        batch_lines.append(batch_entry)

        # Save metadata for evaluation
        metadata[cid] = {
            "question": Q,
            "references": references,
            "candidates": candidates,
            "gold_indices": gold_indices
        }

        count += 1

    # write batch jsonl
    with open(OUTPUT_BATCH_PATH, "w", encoding="utf-8") as f:
        for entry in batch_lines:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(batch_lines)} batch entries to {OUTPUT_BATCH_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")

if __name__ == "__main__":
    create_batch()
