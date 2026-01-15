"""
create_g_eval_batch.py

Create a JSONL batch of prompts for G-Eval style evaluation.
Input:
  - eval_metadata.json   (metadata created by your eval batch creator: contains question, references)
  - eval_batch_results.jsonl  (the model outputs that contain sample answers and selected indices)
Output:
  - g_eval_batch.jsonl   (each line is a batch request object, ready for your /v1/chat/completions batch uploader)

For each metadata entry, we create N_SAMPLES independent evaluation requests
that ask the LLM to rate the provided sample answer vs the gold answer on several criteria.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

# === CONFIG ===
EVAL_METADATA_PATH = Path("eval_metadata_original_webqa_4o.json")  # produced by your create_eval_batch.py
EVAL_BATCH_RESULTS = Path("eval_batch_original_webqa_4o_results.jsonl")  # results from runs (contains sample answers)

# ENHANCED_DATASET_PATH = Path("eval_metadata.json")#("WebQA_enhanced_dataset.json")
# EVAL_BATCH_RESULTS = Path("eval_batch_results.jsonl")

OUTPUT_BATCH = Path("g_eval_batch_original_webqa_4o.jsonl")  # to upload to model
MODEL = "gpt-4.1-mini-2025-04-14" #"gpt-5-mini-2025-08-07" 
N_SAMPLES = 20  # number of LLM samples per GUID (G-Eval uses many draws)
MAX_TOKENS = 200
TEMPERATURE = 1  # To obtain variability for G-Eval

# === SYSTEM PROMPT (strict) ===
SYSTEM_PROMPT = r"""
You are an evaluation assistant. You will be given:
  - A question Q.
  - A gold/reference answer (one or more short reference sentences).
  - A candidate/sample answer produced by a system.

Your job is to **assign integer scores** (1..5) for the following criteria comparing the *candidate* to the *gold/reference*:

1. correctness (1-5): How factually correct is the candidate vs the gold answer? (5 = fully correct, 1 = mostly or fully incorrect).
2. completeness (1-5): How completely does the candidate cover the key points in the gold answer? (5 = covers all key points; 1 = almost none).
3. precision (1-5): How concise and focused is the candidate (not too long answer)? (5 = concise & precise; 1 = verbose / many extraneous claims).
4. coherence (1-5): Is the candidate fluent and well-structured? (5 = natural, coherent; 1 = disfluent or incoherent).
5. relevance (1-5): Are the statements in the candidate relevant to the question and gold answer (not introducing unrelated claims)? (5 = fully relevant; 1 = mostly irrelevant).

REQUIREMENTS (strict):
- Output **ONLY** a JSON object EXACTLY of the form:
  {"correctness": <1-5>, "completeness": <1-5>, "precision": <1-5>, "coherence": <1-5>, "relevance": <1-5>}
- No additional text, explanation, or whitespace outside the JSON object.

FEW-SHOT EXAMPLE (this demonstrates the required output; do not include the example in actual responses):
  Q: "Is Paris the capital of Germany ?"
  Gold: "No, Paris is not the capital of Germany."
  Candidate: "No, the capital of Germany is not Paris, the capital of Germany is Berlin."
  Expected JSON output: {"correctness":5,"completeness":5,"precision":5,"coherence":5,"relevance":5}

  Q: "Is Paris the capital of Germany ?"
  Gold: "No, Paris is not the capital of Germany."
  Candidate: "The capital of Germany is Berlin."
  Expected JSON output: {"correctness":3,"completeness":2,"precision":5,"coherence":5,"relevance":2}

  Q: "Is Paris the capital of Germany ?"
  Gold: "No, Paris is not the capital of Germany."
  Candidate: "If capital stands for the central administrative region of a country, then no, Paris is not the capital of Germany. It's important to note what is the definintion of a 'capital'. However, due to the common popular opinion, a capital of a country is well defined."
  Expected JSON output: {"correctness":5,"completeness":5,"precision":1,"coherence":4,"relevance":1}

END OF SYSTEM PROMPT.
""".strip()


# === helper loaders ===
def load_eval_metadata(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_eval_batch_results(path: Path) -> Dict[str, str]:
    """
    Read eval_batch_results.jsonl and extract the sample 'answer' string for each entry.
    We return a mapping: custom_id -> candidate_answer string
    If there are multiple responses per id in that file, the last one overwrites (you can adapt).
    """
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj.get("custom_id")  # or obj.get("id") or obj.get("customId")
            # candidate extraction: safest effort to parse assistant content JSON
            try:
                body = obj["response"]["body"]
                choice = body["choices"][0]
                content = choice["message"]["content"]
                # content might be a JSON string like '{"selected_indices":[...],"answer":"..."}'
                try:
                    parsed = json.loads(content)
                    candidate_answer = parsed.get("answer", "")
                except Exception:
                    # if content isn't JSON, fall back to the whole text
                    candidate_answer = content
            except Exception:
                candidate_answer = ""
            if cid:
                mapping[cid] = candidate_answer
    return mapping


def make_user_message(
    question: str, gold_refs: List[str], candidate: str
) -> List[Dict[str, Any]]:
    """
    Construct the multimodal-like user message content (list) similar to your batch format.
    It contains the Q, gold references, and the candidate answer.
    """
    msg = []
    msg.append({"type": "text", "text": f"Question: {question}"})
    # join gold reference(s) into a single block
    for i, r in enumerate(gold_refs, start=1):
        msg.append({"type": "text", "text": f"Gold reference {i}: {r}"})
    msg.append({"type": "text", "text": f"Candidate answer: {candidate}"})
    msg.append(
        {
            "type": "text",
            "text": "Please score the candidate exactly with the five integer ratings in JSON as instructed by the system prompt.",
        }
    )
    return msg


def main():
    print("Loading metadata...")
    meta = load_eval_metadata(EVAL_METADATA_PATH)
    print("Loading eval batch results (candidate answers)...")
    cand_map = load_eval_batch_results(EVAL_BATCH_RESULTS)

    out_lines = []
    count = 0
    for guid, entry in meta.items():
        question = entry.get("question", "")
        references = entry.get("references", []) or entry.get("A", []) or []
        candidate_answer = cand_map.get(guid, "")
        if not candidate_answer:
            # skip if no candidate found
            continue

        # Build a single request for this guid that asks the model to return N_SAMPLES completions
        custom_id = f"{guid}_geval"
        user_content = make_user_message(question, references, candidate_answer)

        batch_entry = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "max_tokens": MAX_TOKENS,
                # "max_completion_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "n": N_SAMPLES,  # request N_SAMPLES completions in one API call
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            },
        }
        out_lines.append(batch_entry)
        count += 1

    # write jsonl
    with open(OUTPUT_BATCH, "w", encoding="utf-8") as f:
        for entry in out_lines:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Wrote {count} batch requests to {OUTPUT_BATCH}")


if __name__ == "__main__":
    main()
