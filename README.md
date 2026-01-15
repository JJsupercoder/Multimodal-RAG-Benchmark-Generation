README.txt
Project overview

This repository contains code to (a) build a deterministic multimodal evaluation dataset from WebQA, (b) create batch JSONL requests for LLMs, (c) run those batches against the OpenAI API (or a local runner), and (d) evaluate results with multiple metrics (LLM-based G-Eval, keyword recall, retrieval F1) and reproducibility graphs (CLIP-based). The pipeline is designed to be deterministic (seeded RNG) and reproducible.

Main workflow (high-level)

Typical end-to-end flow:

Build/prepare the dataset / candidate pools (if not already available).

Create a batch JSONL file that will be submitted to the LLM(s).

Optionally estimate API costs for the batch.

Submit the batch to the API (run batch execution).

Collect results and usage logs (responses + openai_usage_log.csv).

Prepare evaluation batches (G-Eval) and/or evaluation metadata.

Submit evaluation batch(s) or evaluate locally.

Compute automatic metrics: recall, retrieval F1, G-Eval aggregation, and reproducibility graphs (CLIP).

Inspect outputs and debug as needed.

Files & short descriptions

Dataset / batch creation

dataset_builder.py
Build or clean the dataset from raw WebQA files. Produces a JSON dataset used downstream (e.g., WebQA_dynamic_dataset_*.json). Run this first if you start from raw data.

dataset_bcreate_batch.py (typo/variant)
Appears to be a variant of dataset/batch building code. Use whichever is the correct/working script in your repo. Both are likely helpers to produce the initial candidate pools or candidate formatting.

create_batch.py
Create the main batch file (sample_batch.jsonl / sample_batch_<mode>.jsonl) used for LLM generation. The script reads the prepared dataset and writes a JSONL batch file where each line is a request entry (custom_id, messages, body...). Output: sample_batch.jsonl (or mode-specific files).

create_eval_batch_original_webqa.py
Create an evaluation batch (likely for the original WebQA baseline). Produces JSONL to send for model evaluation on original WebQA items.

create_eval_batch.py
Create eval batch(s) for your dynamic dataset. Produces JSONL and an evaluation metadata file (e.g., eval_metadata_*.json) containing the candidate ordering and gold_indices used for computing retrieval F1.

run_batch.py
Script that actually runs a JSONL batch file against the API. Reads a batch JSONL (sample_batch.jsonl or eval_batch.jsonl) and submits requests (possibly via OpenAI batch API or sequential calls). Produces a batch_results.jsonl and logs usage to openai_usage_log.csv.

Iog_batch_runs.py (probably log_batch_runs.py)
Utility to parse the openai_usage_log.csv or batch_results.jsonl and produce summary statistics or CSVs. Inspect or rename if needed.

Cost estimation

estimate_batch_cost.py
Estimate the expected token costs and model billing for a batch, given model choices (embedding / completion models), expected token counts or example payloads. Run before run_batch.py if you want to budget.

Evaluation (automatic & LLM-based)

evaluation_create_Geval_batch.py
Create the JSONL batch for G-Eval-style judgments — i.e., prompts that ask an LLM to grade outputs on correctness/completeness/precision/coherence/relevance. Output is a JSONL to be sent to the LLM judge.

evaluation_Geval_score.py
Parse G-Eval results and aggregate per-item criterion scores into final metrics (e.g., mean across criteria, confidence intervals).

evaluation_recall_score.py
Compute the keyword-recall metric between a reference answer and candidate answer. Uses spaCy to extract keywords; output: per-item recall and dataset average recall.

evaluation_retrieval_fl.py
Compute retrieval Precision/Recall/F1 by comparing predicted selected indices (from model output) with gold indices in eval metadata. Produces per-guid F1 and aggregated metrics.

Reproducibility & similarity

reproducibility_graphs_using_clip.py
Compute CLIP text embeddings for Q+A pairs across two runs, compute cosine similarities, compute keyword Jaccard overlaps, and plot ECDF/CCDF graphs. Output: an image like cdf_clip_qapairs.png and numeric summaries.

Misc / logs

openai_usage_log.csv
CSV containing API usage (tokens, cost, model, timestamp). Produced by run_batch.py or logging utilities.


Environment, dependencies, and setup

Recommended: create a virtual environment and install required packages.

Create venv and activate:

python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows (PowerShell: .\.venv\Scripts\Activate.ps1)


Install typical dependencies (adjust if your repo has requirements.txt):

pip install -r requirements.txt 
(# or manually):
pip install simdjson openai transformers torch scikit-learn matplotlib numpy spacy tqdm ddgs


spaCy model (used by evaluation_recall_score.py):

python -m spacy download en_core_web_md


Set OpenAI API key:

export OPENAI_API_KEY="sk-..."
(# or in Windows PowerShell):
setx OPENAI_API_KEY "sk-..."

How to run 
1) Build dataset (if needed)
If you start from raw WebQA files:

python dataset_builder.py


Output: WebQA_dynamic_dataset_*.json

2) Create generation batch
Create the batch that the LLM will answer:

python create_batch.py


Output: sample_batch.jsonl (or mode-specific such as sample_batch_ITT_diff.jsonl) and possibly sample_batch_<mode>_meta.json.

3) Estimate cost (optional)
Estimate API token usage and cost:

python estimate_batch_cost.py --batch sample_batch_ITT_diff.jsonl --model gpt-4o-mini


4) Run the batch (submit to API)
Submit the batch to the OpenAI API (sequential or batch endpoint), producing results and usage logs:

python run_batch.py --input sample_batch_ITT_diff.jsonl --output sample_batch_ITT_diff_results.jsonl


Outputs:

sample_batch_ITT_diff_results.jsonl — model responses

openai_usage_log.csv — token usage and cost summary

5) Create evaluation batch (G-Eval) and metadata
Create an eval batch that asks the model (or judge LLM) to score candidate answers:

python create_eval_batch.py
(# creates: eval_batch_ITT_diff.jsonl, eval_metadata_ITT_diff.json)


6) Run eval batch (judge LLM)
Submit eval_batch_*.jsonl to the judge model (e.g., GPT-4.1/G-Eval):

python run_batch.py --input eval_batch_ITT_diff.jsonl --output eval_batch_ITT_diff_results.jsonl


7) Compute automatic metrics

Keyword recall:

python evaluation_recall_score.py --metadata eval_metadata_ITT_diff.json --results eval_batch_ITT_diff_results.jsonl


Retrieval F1:

python evaluation_retrieval_fl.py --metadata eval_metadata_ITT_diff.json --results eval_batch_ITT_diff_results.jsonl


Aggregate G-Eval:

python evaluation_Geval_score.py --geval_results eval_batch_ITT_diff_results.jsonl


8) Reproducibility graphs (CLIP)
If you have two result runs (run A and run B), compare them:

python reproducibility_graphs_using_clip.py \
  --results1 sample_batch_ITT_diff_results.jsonl \
  --results2 sample_batch_ITT_diff2_results.jsonl \
  --out ecdf_clip_qapairs.png


Output: an ECDF / CCDF plot and numeric stats printed to console.

