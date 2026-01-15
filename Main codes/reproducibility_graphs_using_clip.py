# clip_ecdf_reproducibility.py
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import CLIPProcessor, CLIPModel
from evaluation_recall_score import extract_keywords_from_text  # your function

# ---------- Config ----------
RESULTS_FILE_1 = "sample_batch_ITT_diff_results.jsonl"
RESULTS_FILE_2 = "sample_batch_ITT_diff2_results.jsonl"
OUT_PLOT = "ecdf_clip_qapairs.png"
BATCH_SIZE = 64          # batch size for encoding text with CLIP
CLIP_MODEL = "openai/clip-vit-base-patch32"
MAX_LENGTH = 77          # CLIP text encoder max tokens (keep <= model config max_position_embeddings)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------

# ---- Utilities ----
def load_result_batch(file_path):
    """Return dict custom_id -> {'question':..., 'answer':...} for lines with both markers."""
    data = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                entry = json.loads(ln)
            except Exception:
                continue
            # robustly fetch the message content (common shape)
            content = None
            try:
                content = entry.get("response", {}).get("body", {}).get("choices", [])[0].get("message", {}).get("content")
            except Exception:
                pass
            if not content:
                content = entry.get("response", {}).get("body", {}).get("output") \
                          or entry.get("response", {}).get("output") \
                          or entry.get("output") \
                          or entry.get("content")
            if not content or not isinstance(content, str):
                continue
            if "Question:" in content and "Answer:" in content:
                try:
                    q_part, a_part = content.split("Answer:", 1)
                    question = q_part.replace("Question:", "").strip()
                    answer = a_part.strip()
                except Exception:
                    continue
                cid = entry.get("custom_id") or entry.get("customId") or entry.get("id")
                if cid:
                    data[str(cid)] = {"question": question, "answer": answer}
    return data

def make_text_pair_strings(data, ids):
    """Return list of strings for ids: question + ' ' + answer (short)."""
    out = []
    for cid in ids:
        q = data[cid]["question"].strip()
        a = data[cid]["answer"].strip()
        combined = (q + " " + a).strip()
        # optional: truncate combined string to avoid huge inputs (CLIP has limited token length)
        out.append(combined)
    return out

# ---- CLIP encoding ----
def init_clip(model_name=CLIP_MODEL, device=DEVICE):
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    # set model to eval
    model.eval()
    return model, processor

def encode_texts_clip(model, processor, texts, batch_size=BATCH_SIZE, device=DEVICE, max_length=MAX_LENGTH):
    """
    Encode a list of texts using CLIP text encoder.
    Returns numpy array shape (N, D) normalized (L2).
    """
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            # processor will handle tokenization; use truncation + padding
            inputs = processor(text=batch_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
            # move tensors to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # get text features
            outputs = model.get_text_features(**inputs)  # (B, D)
            # convert to CPU numpy
            emb = outputs.cpu().numpy()
            # normalize rows
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            emb = emb / norms
            all_embs.append(emb)
    if all_embs:
        return np.vstack(all_embs)
    else:
        return np.zeros((0, model.text_projection.shape[1] if hasattr(model, 'text_projection') else model.config.projection_dim))

def cosine_sim_between_embeddings(a, b):
    """Compute cosine similarities elementwise between two equal-length arrays."""
    # a, b: (N, D)
    sims = np.sum(a * b, axis=1)
    return sims

# ---- Keyword overlap (Jaccard) ----
def compute_keyword_overlap_for_pairs(data1, data2, ids):
    overlaps = []
    for cid in ids:
        t1 = (data1[cid]["question"] + " " + data1[cid]["answer"]).strip()
        t2 = (data2[cid]["question"] + " " + data2[cid]["answer"]).strip()
        kws1 = extract_keywords_from_text(t1.lower())
        kws2 = extract_keywords_from_text(t2.lower())
        if not kws1 and not kws2:
            overlaps.append(1.0)  # if both empty, treat as identical
        else:
            union = kws1 | kws2
            if len(union) == 0:
                overlaps.append(0.0)
            else:
                overlaps.append(len(kws1 & kws2) / len(union))
    return overlaps

# ---- ECDF plotting ----
def plot_ecdf(ax, values, title, color=None):
    if len(values) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(title)
        return
    vals = np.sort(np.asarray(values))
    n = len(vals)
    y = np.arange(1, n+1) / n
    ax.step(vals, y, where='post', label=f"n={n}", color=color)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(title)
    ax.set_ylabel("ECDF")
    ax.grid(alpha=0.3)
    ax.legend()

# ---- Main flow ----
def main():
    print(f"Device: {DEVICE}")
    data1 = load_result_batch(RESULTS_FILE_1)
    data2 = load_result_batch(RESULTS_FILE_2)
    matched = sorted(set(data1.keys()) & set(data2.keys()))
    print(f"Matched items: {len(matched)}")

    if len(matched) == 0:
        print("No matched IDs between runs; exiting.")
        return

    # prepare texts
    texts1 = make_text_pair_strings(data1, matched)
    texts2 = make_text_pair_strings(data2, matched)
    # initialize CLIP
    model, processor = init_clip()

    # encode with CLIP (concatenate both lists but we need to preserve order)
    # to be memory-efficient, encode separately
    emb1 = encode_texts_clip(model, processor, texts1, batch_size=BATCH_SIZE, device=DEVICE, max_length=MAX_LENGTH)
    emb2 = encode_texts_clip(model, processor, texts2, batch_size=BATCH_SIZE, device=DEVICE, max_length=MAX_LENGTH)

    # ensure shapes
    assert emb1.shape[0] == len(matched)
    assert emb2.shape[0] == len(matched)

    # cosine similarities
    cosine_scores = cosine_sim_between_embeddings(emb1, emb2)  # array length = matched
    # clip numeric safety to [-1,1]
    cosine_scores = np.clip(cosine_scores, -1.0, 1.0).tolist()

    # keyword overlaps
    kw_overlaps = compute_keyword_overlap_for_pairs(data1, data2, matched)

    # stats helper
    def stats(arr):
        if not arr:
            return {}
        a = np.asarray(arr)
        return {
            "n": int(a.size),
            "mean": float(a.mean()),
            "std": float(a.std(ddof=0)),
            "median": float(np.median(a)),
            "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)),
            "frac_exact_1": float((a == 1.0).sum() / a.size)
        }

    cos_stats = stats(cosine_scores)
    kw_stats = stats(kw_overlaps)

    print("\n=== Cosine similarity stats (CLIP embeddings) ===")
    for k, v in cos_stats.items():
        print(f"{k}: {v}")

    print("\n=== Keyword overlap stats ===")
    for k, v in kw_stats.items():
        print(f"{k}: {v}")

    # Plot ECDFs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plot_ecdf(ax1, cosine_scores, "Cosine similarity (CLIP)")
    plot_ecdf(ax2, kw_overlaps, "Keyword overlap (Jaccard)")
    fig.suptitle("ECDF: reproducibility between two runs (CLIP embeddings + keywords)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUT_PLOT, dpi=200)
    print(f"Saved ECDF plot to {OUT_PLOT}")
    plt.show()

if __name__ == "__main__":
    main()
