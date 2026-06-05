README.txt
# Project overview

This repository contains code to (a) build a deterministic multimodal evaluation dataset from WebQA, (b) create batch JSONL requests for the ChatGPT Batch API, (c) run those batches using the ChatGPT Batch API  and (d) evaluate results with multiple metrics (LLM-based G-Eval, keyword recall, retrieval F1) and reproducibility graphs (CLIP-based). The pipeline is designed to be deterministic (seeded RNG) and reproducible by using a set of parameters.

# Main workflow (high-level):
1. Build/prepare the dataset / candidate pools (if not already available).
2. Create a batch JSONL file that will be submitted to the ChatGPT Batch API.
3. Optionally estimate API costs for the batch.
4. Submit the batch to the API (run batch execution).
5. Collect the generated benchmark dataset and usage logs (responses + openai_usage_log.csv).
6. Prepare evaluation batches (G-Eval) and/or evaluation metadata.
7. Submit evaluation batch(s).
8. Compute automatic metrics: recall, retrieval F1, G-Eval aggregation, and reproducibility graphs (CLIP).
9. Provide insights into the reproducibility of the generation of the benchmark with the same parameters.

# Files & short descriptions -
## Dataset / batch creation

1. dataset_builder.py
Build or clean the dataset from raw WebQA files. Produces a JSON dataset used downstream (e.g., WebQA_dynamic_dataset_*.json). Run this first if you start from raw data.

2. create_batch.py
Create the main batch file (sample_batch.jsonl / sample_batch_<mode>.jsonl) used for LLM generation. The script reads the prepared dataset and writes a JSONL batch file where each line is a request entry (custom_id, messages, body...). Output: sample_batch.jsonl (or mode-specific files).

3. create_eval_batch_original_webqa.py
Create an evaluation batch (likely for the original WebQA baseline). Produces JSONL to send for model evaluation on original WebQA items.

4. create_eval_batch.py
Create eval batch(s) for your dynamic dataset. Produces JSONL and an evaluation metadata file (e.g., eval_metadata_*.json) containing the candidate ordering and gold_indices used for computing retrieval F1.

5. run_batch.py
Script that actually runs a JSONL batch file against the API. Reads a batch JSONL (sample_batch.jsonl or eval_batch.jsonl) and submits requests (possibly via OpenAI batch API or sequential calls). Produces a batch_results.jsonl and logs usage to openai_usage_log.csv.

6. Iog_batch_runs.py (probably log_batch_runs.py)
Utility to parse the openai_usage_log.csv or batch_results.jsonl and produce summary statistics or CSVs. Inspect or rename if needed.

## Cost estimation

1. estimate_batch_cost.py
Estimate the expected token costs and model billing for a batch, given model choices (embedding / completion models), expected token counts or example payloads. Run before run_batch.py if you want to budget.

## Evaluation (automatic & LLM-based)

1. evaluation_create_Geval_batch.py
Create the JSONL batch for G-Eval-style judgments — i.e., prompts that ask an LLM to grade outputs on correctness/completeness/precision/coherence/relevance. Output is a JSONL to be sent to the LLM judge.

2. evaluation_Geval_score.py
Parse G-Eval results and aggregate per-item criterion scores into final metrics (e.g., mean across criteria, confidence intervals).

3. evaluation_recall_score.py
Compute the keyword-recall metric between a reference answer and candidate answer. Uses spaCy to extract keywords; output: per-item recall and dataset average recall.

4. evaluation_retrieval_fl.py
Compute retrieval Precision/Recall/F1 by comparing predicted selected indices (from model output) with gold indices in eval metadata. Produces per-guid F1 and aggregated metrics.

## Reproducibility & similarity

1. reproducibility_graphs_using_clip.py
Compute CLIP text embeddings for Q+A pairs across two runs, compute cosine similarities, compute keyword Jaccard overlaps, and plot ECDF/CCDF graphs. Output: an image like cdf_clip_qapairs.png and numeric summaries.

## Misc / logs
1. openai_usage_log.csv
2. CSV containing API usage (tokens, cost, model, timestamp). Produced by run_batch.py or logging utilities.
3. Environment, dependencies, and setup

# How to run -
## Initial Steps
Recommended: create a virtual environment and install required packages.

1. Create venv and activate:

python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows (PowerShell: .\.venv\Scripts\Activate.ps1)


2. Install typical dependencies (adjust if your repo has requirements.txt):

pip install -r requirements.txt 
(# or manually):
pip install simdjson openai transformers torch scikit-learn matplotlib numpy spacy tqdm ddgs

for spaCy model (used by evaluation_recall_score.py):
python -m spacy download en_core_web_md


3. Set OpenAI API key:

export OPENAI_API_KEY="sk-..."
(# or in Windows PowerShell):
setx OPENAI_API_KEY "sk-..."

4. Download the original WebQA dataset and paste it in the root directory -
https://webqna.github.io/

## Main steps
### Easy way -
CMD: python main.py
This actually does the below steps, except for producing the reproducibility graph, for which you need to compare 2 consecutive runs with the same parameters. 

### Detailed way -
1. CMD: python dataset_builder.py
Output: WebQA_dynamic_dataset_*.json

2. Create the dataset generation batch
Create the QA dataset batch that the LLM will be tested on-
CMD: python create_batch.py
Output: sample_batch.jsonl (or mode-specific such as sample_batch_ITT_diff.jsonl) and possibly sample_batch_<mode>_meta.json.

3) Estimate cost (optional)
Estimate API token usage and cost
CMD: python estimate_batch_cost.py --batch sample_batch_<mode>.jsonl --model gpt-4o-mini

4) Run the batch (submit to API)
Submit the batch to the OpenAI API (sequential or batch endpoint), producing results and usage logs
CMD: python run_batch.py --input sample_batch_<mode>.jsonl --output sample_batch_<mode>_results.jsonl
Outputs:
sample_batch_ITT_diff_results.jsonl — model responses
openai_usage_log.csv — token usage and cost summary

5) Create evaluation batch (G-Eval) and metadata
Create an eval batch that asks the model (or judge LLM) to score candidate answers
CMD: python create_eval_batch.py
Outputs:
eval_batch_<mode>.jsonl
eval_metadata_<mode>.json

6) Run eval batch (judge LLM)
Submit eval_batch_<mode>.jsonl to the judge model (e.g., GPT-4.1/G-Eval)
CMD: python run_batch.py --input eval_batch_<mode>.jsonl --output eval_batch_<mode>_results.jsonl

7) Compute automatic metrics -

Keyword recall:
CMD: python evaluation_recall_score.py --metadata eval_metadata_<mode>.json --results eval_batch_<mode>_results.jsonl

Retrieval F1:
CMD: python evaluation_retrieval_fl.py --metadata eval_metadata_<mode>.json --results eval_batch_<mode>_results.jsonl

Aggregate G-Eval:
python evaluation_Geval_score.py --geval_results eval_batch_<mode>_results.jsonl

8) Reproducibility graphs (CLIP)
If you have two result runs (run A and run B), compare them:

CMD: python reproducibility_graphs_using_clip.py \
  --results1 sample_batch_ITT_diff_results.jsonl \
  --results2 sample_batch_ITT_diff2_results.jsonl \
  --out ecdf_clip_qapairs.png

Output: an ECDF / CCDF plot and numeric stats printed to console.

