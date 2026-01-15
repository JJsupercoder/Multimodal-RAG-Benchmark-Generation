# run_openai_batch_and_log.py
def exec_batch(
    batch_input_file="sample_batch.jsonl",
    batch_result_file="batch_results.jsonl",
    batch_error_file="batch_errors.jsonl",
):
    import openai
    import time
    import os
    import json
    import csv
    from datetime import datetime
    from dotenv import load_dotenv

    # === Load API Key ===
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)

    # === Step 1: Upload file ===
    print("Uploading batch file...")
    upload = client.files.create(file=open(batch_input_file, "rb"), purpose="batch")
    file_id = upload.id
    print(f"Uploaded file ID: {file_id}")

    # === Step 2: Create batch job ===
    print("Creating batch job...")
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"purpose": "Testing the quality of questions"},
    )
    batch_id = batch.id
    print(f"Batch job ID: {batch_id}")

    with open("save_file_id.txt", "a") as save_file:
        save_file.writelines(
            [f"\nUploaded file ID: {file_id}", f"\nBatch job ID: {batch_id}\n"]
        )

    # === Step 3: Wait for job to complete with progress tracking ===
    print("Please be patient as it takes time to complete.")
    print("Waiting for batch job to complete...")
    while True:
        status = client.batches.retrieve(batch_id)

        total_requests = status.request_counts.total
        completed_requests = status.request_counts.completed
        failed_requests = status.request_counts.failed

        percent = (
            ((completed_requests + failed_requests) / total_requests) * 100
            if total_requests
            else 0
        )
        print(f"Status: {status.status} | Completed: ({percent:.2f}%)")

        if status.status in ("completed", "failed", "cancelled", "expired"):
            break
        time.sleep(60)

    # status = client.batches.retrieve("batch_684c2b87d58c819095683c306c64043d")

    result_file_id = status.output_file_id
    error_file_id = status.error_file_id
    print(status)
    print("OpenAI Result file id:", result_file_id)
    print("OpenAI Error id:", error_file_id)

    # === Step 4: Download result file ===
    # result_file_id = status.output_file_id
    # result_file = client.files.retrieve(result_file_id)
    # result_content = client.files.content(result_file.id)

    result_content = client.files.content(result_file_id)

    # save the batch result file locally
    with open(batch_result_file, "w", encoding="utf-8") as f:
        for chunk in result_content.iter_lines():
            f.write(chunk + "\n")
    print(f"Batch results saved to: {batch_result_file}")

    # save the batch error file locally
    if error_file_id is not None:
        error_content = client.files.content(error_file_id)
        with open(batch_error_file, "w", encoding="utf-8") as f:
            for chunk in error_content.iter_lines():
                f.write(chunk + "\n")
        print(f"Batch results saved to: {batch_error_file}")


if __name__ == "__main__":
    # batch_input_file = "sample_batch.jsonl"
    # batch_result_file = "batch_results.jsonl"
    # batch_error_file = "batch_errors.jsonl"
    batch_input_file = "g_eval_batch_original_webqa.jsonl"
    batch_result_file = "g_eval_batch_original_webqa_results.jsonl"
    batch_error_file = "g_eval_batch_original_webqa_errors.jsonl"
    exec_batch(batch_input_file, batch_result_file, batch_error_file)
