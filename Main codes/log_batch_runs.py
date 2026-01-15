import json
import csv
from datetime import datetime

# Pricing per model (per 1M tokens)
MODEL_PRICING = {
    "gpt-4.1-mini-2025-04-14": {"prompt": 0.40, "completion": 1.60},
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
}

def log_batch_summary(batch_result_file, csv_file_path):
    total_prompt_tokens = 0
    total_completion_tokens = 0

    with open(batch_result_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            response = data["response"]
            body = response['body']
            usage = body["usage"]
            model = body.get("model")

            total_prompt_tokens += usage["prompt_tokens"]
            total_completion_tokens += usage["completion_tokens"]

    if model not in MODEL_PRICING:
        raise ValueError(f"Unknown model: {model}")

    prompt_cost = (total_prompt_tokens / 1e6) * MODEL_PRICING[model]["prompt"]
    completion_cost = (total_completion_tokens / 1e6) * MODEL_PRICING[model]["completion"]
    total_cost = prompt_cost + completion_cost
    total_tokens = total_prompt_tokens + total_completion_tokens

    # Append summary to CSV
    with open(csv_file_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now(),
            total_prompt_tokens,
            total_completion_tokens,
            total_tokens,
            round(total_cost, 6),
            "batch api",
            model
        ])

    print(f"Logged batch usage: {total_tokens} tokens → ${total_cost:.6f}")

if __name__ == "__main__":
    log_batch_summary("evaluation_g_eval_original_batch_results.jsonl", "openai_usage_log.csv")
