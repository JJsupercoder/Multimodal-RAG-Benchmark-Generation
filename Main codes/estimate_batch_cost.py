# estimate_batch_cost.py
def calc_batch_cost():
    import tiktoken
    import json

    BATCH_FILE = "sample_batch.jsonl"
    MODEL = "gpt-4.1-mini-2025-04-14"

    # GPT-4.1 Mini pricing (Batch API): prices per 1M tokens
    INPUT_COST_PER_1M = 0.40
    OUTPUT_COST_PER_1M = 1.60
    EXPECTED_OUTPUT_TOKENS = 1000  # or change to match your max_tokens

    # Encoding (GPT-4.1 Mini uses cl100k_base like GPT-4 Turbo/4o)
    encoding = tiktoken.get_encoding("cl100k_base")

    total_input_tokens = 0
    total_output_tokens = 0

    with open(BATCH_FILE, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            messages = obj["body"]["messages"]
            input_text = ""
            for msg in messages:
                content = msg["content"]
                if isinstance(content, list):
                    for part in content:
                        if part["type"] == "text":
                            input_text += part["text"] + "\n"
                elif isinstance(content, str):
                    input_text += content + "\n"

            input_tokens = len(encoding.encode(input_text))
            total_input_tokens += input_tokens
            total_output_tokens += EXPECTED_OUTPUT_TOKENS

    est_cost = (total_input_tokens / 1e6) * INPUT_COST_PER_1M + (total_output_tokens / 1e6) * OUTPUT_COST_PER_1M

    print(f"Total estimated input tokens: {total_input_tokens}")
    print(f"Total estimated output tokens: {total_output_tokens}")
    print(f"Estimated total cost: ${est_cost:.6f}")

if __name__ == "__main__":
    calc_batch_cost()
