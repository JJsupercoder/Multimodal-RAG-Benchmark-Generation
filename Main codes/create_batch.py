#!/usr/bin/env python3
"""
create_batch.py (modified)
- Builds batch JSONL (same format as before) AND concurrently constructs a WebQA_dynamic_dataset.json
  containing only: webqa_id, img_posFacts, txt_posFacts, img_negFacts, txt_negFacts (<= 12 sources total).
- Negative selection: prefer retrieved txtnegFacts first, then entry txt_negFacts; only if both insufficient,
  fill the remainder from img_negFacts.
Minimal changes to your original code; retains your system prompt and batch formatting.
"""

import simdjson
import json
import random
from select_additional_source import retrieve_add_source
# from main import SEED
from select_additional_source import SourceModeNotFoundError
from select_additional_source import WikiURLNotFoundError
from pathlib import Path

# === CONFIG === (you can adjust these as before)
INPUT_PATH_DEFAULT = r"D:\MSc AI\Sem 2\Dissertation\WebQA_data_first_release\WebQA_train_val.json"
OUTPUT_BATCH_DEFAULT = r"D:\MSc AI\Sem 2\Dissertation\Code v2\sample_batch.jsonl"
OUTPUT_DYNAMIC_DATASET = r"D:\MSc AI\Sem 2\Dissertation\Code v2\WebQA_dynamic_dataset.json"

def create_batch_jsonl(input_path, output_path, system_prompt="You are a helpful question generator.",
                       model="gpt-4.1-mini-2025-04-14", n=100, random_sample=True, seed=42, mode="IT"):
    parser = simdjson.Parser()
    with open(input_path, "rb") as f:
        data = parser.parse(f.read(), recursive=True)

    # counters for synthetic ids
    image_id_counter = 90_000_000
    def next_image_id():
        nonlocal image_id_counter
        image_id_counter += 1
        return image_id_counter

    def next_text_id(guid, idx):
        return f"{guid}_txt_{idx}"

    # output dataset accumulation
    dynamic_dataset = {}

    with open(output_path, "w", encoding="utf-8") as out_file:
        all_keys = list(data.keys())[:2000]  # same as before
        if random_sample:
            random.seed(seed)
            random.shuffle(all_keys)

        entry_count = 0
        iter_var = 0
        while entry_count < n:
            if entry_count % 10 == 0 and entry_count != 0:
                print(entry_count, "entries done")
            if iter_var >= len(all_keys):
                break
            guid = all_keys[iter_var]
            iter_var += 1
            entry = data[guid]
            img_posFacts = entry.get("img_posFacts", [])
            txt_posFacts = entry.get("txt_posFacts", [])
            img_negFacts_entry = entry.get("img_negFacts", [])
            txt_negFacts_entry = entry.get("txt_negFacts", [])
            example_question = entry.get("Q")
            example_answer = entry.get("A", [""])[0]
            extra = None

            if img_posFacts == [] and txt_posFacts == []:
                # nothing to work with
                print("Skipping (no sources):", guid)
                continue

            # Ensure mode constraints (kept your checks)
            if ((mode == "IT" or mode == "ITT_same") and not (txt_posFacts == [] and img_posFacts != [] and img_negFacts_entry != [])):
                # print("Exit1", mode, txt_posFacts == [], img_posFacts != [], img_negFacts_entry != [])
                continue

            if ((mode == "IIT" or mode == "ITT_diff") and not (txt_posFacts == [] and len(img_posFacts) >= 2 and img_negFacts_entry != [])):
                # print("Exit2")
                continue

            # Initialize positive lists to send to model (and add to dynamic dataset)
            new_img_posFacts = []
            new_txt_posFacts = []
            # negative pools that we will fill according to mode-specific targets
            chosen_img_neg = []
            chosen_txt_neg = []

            # Temporary holders for retrieved negatives/positives from retrieve_add_source
            retrieved_txtpos = []
            retrieved_txtneg = []

            try:
                # Build positive facts per mode by calling retrieve_add_source as before
                if mode == "IT":
                    img1 = random.choice(img_posFacts)
                    img1_link = img1["imgUrl"]
                    img1_title = img1.get("title", "")
                    # retrieve; may return (txtposFacts, txtnegFacts, pos_txt_title, pos_txt_Url)
                    txtposFacts, txtnegFacts, pos_txt_title, pos_txt_Url = retrieve_add_source(img1_link, "", img1_title, "", mode)
                    retrieved_txtpos = txtposFacts or []
                    retrieved_txtneg = txtnegFacts or []

                    img_fact_dict = {
                        "title": img1_title,
                        "caption": img1.get("caption", ""),
                        "url": img1.get("url", ""),
                        "imgUrl": img1_link
                    }
                    new_img_posFacts.append(img_fact_dict)

                    if retrieved_txtpos:
                        txt_fact = {"title": pos_txt_title or "", "fact": retrieved_txtpos[0], "url": pos_txt_Url or ""}
                        new_txt_posFacts.append(txt_fact)


                elif mode == "ITT_same":
                    img1 = random.choice(img_posFacts)
                    img1_link = img1["imgUrl"]
                    img1_title = img1.get("title", "")
                    txtposFacts, txtnegFacts, pos_txt_title, pos_txt_Url = retrieve_add_source(img1_link, "", img1_title, "", mode)
                    retrieved_txtpos = txtposFacts or []
                    retrieved_txtneg = txtnegFacts or []

                    img_fact_dict = {
                        "title": img1_title,
                        "caption": img1.get("caption", ""),
                        "url": img1.get("url", ""),
                        "imgUrl": img1_link
                    }
                    new_img_posFacts.append(img_fact_dict)
                    
                    chosen_texts = []
                    if len(retrieved_txtpos) >= 2:
                        chosen_texts = retrieved_txtpos[:2]
                    else:
                        if retrieved_txtpos:
                            chosen_texts.extend(retrieved_txtpos)
                        # try entry pos facts:
                        for t in txt_posFacts:
                            if len(chosen_texts) >= 2:
                                break
                            chosen_texts.append(t.get("fact") or t.get("text") or "")
                    if len(chosen_texts) < 2:
                        continue
                    new_txt_posFacts.append({"title": pos_txt_title or "", "fact": chosen_texts[0], "url": pos_txt_Url or ""})
                    new_txt_posFacts.append({"title": pos_txt_title or "", "fact": chosen_texts[1], "url": pos_txt_Url or ""})

                elif mode == "ITT_diff":
                    # two images, two different wiki text pos facts (retrieved)
                    if len(img_posFacts) < 2:
                        continue
                    img1 = img_posFacts[0]; img2 = img_posFacts[1]
                    img1_link, img2_link = img1.get("imgUrl",""), img2.get("imgUrl","")
                    txtposFacts, txtnegFacts, pos_txt1_title, pos_txt2_title, pos_txt1_Url, pos_txt2_Url = retrieve_add_source(img1_link, img2_link, img1.get("title",""), img2.get("title",""), mode)
                    retrieved_txtpos = txtposFacts or []
                    retrieved_txtneg = txtnegFacts or []

                    img1_fact_dict = {"title": img1.get("title",""), "caption": img1.get("caption",""), "url": img1.get("url",""), "imgUrl": img1_link}
                    new_img_posFacts.append(img1_fact_dict)
                    # text pos should include two items (from two wikis)
                    if len(retrieved_txtpos) >= 2:
                        new_txt_posFacts.append({"title": pos_txt1_title or "", "fact": retrieved_txtpos[0], "url": pos_txt1_Url or ""})
                        new_txt_posFacts.append({"title": pos_txt2_title or "", "fact": retrieved_txtpos[1], "url": pos_txt2_Url or ""})
                    else:
                        # fallback to entry txt_posFacts and/or retrieved
                        chosen_texts = []
                        if retrieved_txtpos:
                            chosen_texts.extend(retrieved_txtpos)
                        for t in txt_posFacts:
                            if len(chosen_texts) >= 2:
                                break
                            chosen_texts.append(t.get("fact") or t.get("text") or "")
                        if len(chosen_texts) < 2:
                            continue
                        new_txt_posFacts.append({"title": "", "fact": chosen_texts[0], "url": ""})
                        new_txt_posFacts.append({"title": "", "fact": chosen_texts[1], "url": ""})

                elif mode == "IIT":
                    if len(img_posFacts) < 2:
                        continue
                    img1 = img_posFacts[0]; img2 = img_posFacts[1]
                    img1_link, img2_link = img1.get("imgUrl",""), img2.get("imgUrl","")
                    txtposFacts, txtnegFacts, pos_txt_title, pos_txt_Url = retrieve_add_source(img1_link, img2_link, img1.get("title",""), img2.get("title",""), mode)
                    retrieved_txtpos = txtposFacts or []
                    retrieved_txtneg = txtnegFacts or []

                    img1_fact_dict = {"title": img1.get("title",""), "caption": img1.get("caption",""), "url": img1.get("url",""), "imgUrl": img1_link}
                    img2_fact_dict = {"title": img2.get("title",""), "caption": img2.get("caption",""), "url": img2.get("url",""), "imgUrl": img2_link}
                    new_img_posFacts.extend([img1_fact_dict, img2_fact_dict])

                    # one text pos (prefer entry txt_posFacts then retrieved)
                    if txt_posFacts:
                        t0 = txt_posFacts[0]
                        new_txt_posFacts.append({"title": t0.get("title",""), "fact": t0.get("fact") or t0.get("text",""), "url": t0.get("url","")})
                    elif retrieved_txtpos:
                        new_txt_posFacts.append({"title": pos_txt_title or "", "fact": retrieved_txtpos[0], "url": pos_txt_Url or ""})
                    else:
                        continue
                else:
                    raise SourceModeNotFoundError(mode)
            except WikiURLNotFoundError as w:
                # fallback: skip this guid if retrieval failed and we cannot create pos facts
                print("WikiURLNotFoundError for guid", guid, "; skipping.")
                continue
            except SourceModeNotFoundError:
                print("SourceModeNotFoundError for mode", mode)
                continue
            except Exception as e:
                print("Exception while processing guid", guid, e)
                continue

            # At this point we have new_img_posFacts and new_txt_posFacts (positives)
            # Build negative pools:
            # retrieved_txtneg (from retrieve_add_source) prioritized,
            # then entry txt_negFacts, then entry img_negFacts as last resort
            # convert pools to normalized dict forms
            def normalize_img_item(it):
                return {
                    "image_id": it.get("image_id") if it.get("image_id") else None,
                    "title": it.get("title") or it.get("caption") or "",
                    "caption": it.get("caption") or it.get("title") or "",
                    "url": it.get("url") or "",
                    "imgUrl": it.get("imgUrl") or it.get("image_url") or ""
                }

            def normalize_txt_item(it):
                # it can be dict or string (retrieved lists might be strings)
                if isinstance(it, dict):
                    return {
                        "title": it.get("title",""),
                        "fact": it.get("fact") or it.get("text") or "",
                        "url": it.get("url",""),
                        "text_id": it.get("snippet_id") or it.get("text_id") or None
                    }
                else:
                    # string
                    return {"title": "", "fact": str(it), "url": "", "text_id": None}

            txt_neg_pool = []
            # preference: retrieved_txtneg first (if present)
            if retrieved_txtneg:
                for t in retrieved_txtneg:
                    txt_neg_pool.append(normalize_txt_item(t))
            # then entry-level txt_negFacts
            if txt_negFacts_entry:
                for t in txt_negFacts_entry:
                    txt_neg_pool.append(normalize_txt_item(t))
            # dedupe text by fact
            seen_texts = set()
            dedup_txt_neg = []
            for t in txt_neg_pool:
                key = (t.get("fact","") or "").strip()
                if not key:
                    continue
                if key in seen_texts:
                    continue
                seen_texts.add(key)
                dedup_txt_neg.append(t)
            txt_neg_pool = dedup_txt_neg

            # image negative pool simply from entry (we will use as last resort)
            img_neg_pool = [normalize_img_item(im) for im in img_negFacts_entry]
            # dedupe by imgUrl
            seen_imgurls = set()
            dedup_img_neg = []
            for im in img_neg_pool:
                key = (im.get("imgUrl","") or "").strip()
                if not key:
                    continue
                if key in seen_imgurls:
                    continue
                seen_imgurls.add(key)
                dedup_img_neg.append(im)
            img_neg_pool = dedup_img_neg

            # Determine negative targets by mode (to have total sources <= 12)
            pos_count = len(new_img_posFacts) + len(new_txt_posFacts)
            total_needed = 12 - pos_count
            if mode == "IT":
                target_txt_neg = 5
                target_img_neg = 5
            elif mode in ("ITT_same", "ITT_diff"):
                target_txt_neg = 4
                target_img_neg = 5
            elif mode == "IIT":
                target_txt_neg = 5
                target_img_neg = 4
            else:
                target_txt_neg = total_needed // 2
                target_img_neg = total_needed - target_txt_neg

            # Pick text negatives prioritizing the txt_neg_pool (retrieved first then entry)
            take_txt = min(target_txt_neg, len(txt_neg_pool))
            chosen_txt_neg = txt_neg_pool[:take_txt]

            # If txt negatives are insufficient but there are still total slots, attempt to take more text
            # from any remaining txt_neg_pool entries (already covered); otherwise eventually use img negatives
            cur_neg_count = len(chosen_txt_neg)

            # now pick image negatives up to target_img_neg
            take_img = min(target_img_neg, len(img_neg_pool))
            chosen_img_neg = img_neg_pool[:take_img]

            cur_neg_count = len(chosen_txt_neg) + len(chosen_img_neg)

            # If still fewer than total_needed, try to add remaining text negatives (maybe target_txt_neg < total_needed)
            if cur_neg_count < total_needed:
                need_more = total_needed - cur_neg_count
                # try to take more text negatives (if any left)
                remaining_txt_candidates = txt_neg_pool[take_txt:]
                add_txt_more = remaining_txt_candidates[:need_more]
                chosen_txt_neg.extend(add_txt_more)
                cur_neg_count = len(chosen_txt_neg) + len(chosen_img_neg)

            # If still underfilled, take more images
            if cur_neg_count < total_needed:
                need_more = total_needed - cur_neg_count
                remaining_img_candidates = img_neg_pool[take_img:]
                add_img_more = remaining_img_candidates[:need_more]
                chosen_img_neg.extend(add_img_more)
                cur_neg_count = len(chosen_txt_neg) + len(chosen_img_neg)

            # final safety: if still less than needed, accept fewer negatives (no more available)
            # ensure each chosen image has an image_id, and text negs have text_id
            for im in chosen_img_neg:
                if not im.get("image_id"):
                    im["image_id"] = next_image_id()
            # text ids
            for idx, t in enumerate(chosen_txt_neg):
                if not t.get("text_id"):
                    t["text_id"] = next_text_id(guid, idx)

            # SANITIZE positives too: ensure image ids and text ids exist
            for im in new_img_posFacts:
                if not im.get("image_id"):
                    im["image_id"] = next_image_id()
            for idx, t in enumerate(new_txt_posFacts):
                if not t.get("text_id"):
                    t["text_id"] = next_text_id(guid, idx)

            # Build dynamic dataset entry
            dyn_entry = {
                "webqa_id": guid,
                "img_posFacts": new_img_posFacts,
                "txt_posFacts": new_txt_posFacts,
                "img_negFacts": chosen_img_neg,
                "txt_negFacts": chosen_txt_neg
            }
            dynamic_dataset[guid] = dyn_entry

            # Build user_content for batch (same as your original code)
            user_content = []
            # Add image sources
            for i, img in enumerate(new_img_posFacts):
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": img["imgUrl"],
                        "detail": "low"
                    }
                })
                caption_text = f"Caption for image source {i + 1}: {img.get('caption', 'No caption provided')}."
                user_content.append({
                    "type": "text",
                    "text": caption_text
                })

            # Add text sources
            for fact in new_txt_posFacts:
                user_content.append({
                    "type": "text",
                    "text": f"Text source: {fact['fact']} (from: {fact.get('title','')})"
                })

            batch_entry = {
                "custom_id": guid,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "temperature": 0.0,
                    "top_p": 0.0,
                    "max_tokens": 2000,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                }
            }

            out_file.write(json.dumps(batch_entry) + "\n")

            entry_count += 1

    # Write dynamic dataset to file
    with open(OUTPUT_DYNAMIC_DATASET, "w", encoding="utf-8") as f_dyn:
        json.dump(dynamic_dataset, f_dyn, indent=2, ensure_ascii=False)

    print(f"JSONL saved to {output_path}")
    print(f"Dynamic dataset saved to {OUTPUT_DYNAMIC_DATASET} with {len(dynamic_dataset)} GUIDs.")


system_prompt = '''
# Identity

You are an intelligent question-answer generator with the capability to understand and reason on texts and images.
You generate simple but intelligent questions and the appropriate answers based only on the text and image sources provided to you.

# Instructions

* You will be given some images along with its captions and some text sources. You have to create a new question based on all the image and text sources given to you. 
* You need to think critically and step by step to find 1 or 2 common aspects that can be used to connect all the sources. Those common aspects would be the basis for your new question. 
* Your question should require multimodal reasoning to derive the answer, i.e., requiring all of the text and image sources to be answerable, and the question should not require any other external information besides the provided sources.
* After creating the question you come up with the answer which is based on all the sources and only requires those sources. 
* The question should not refer to any of the sources directly, like mentioning 'In which of the above images...', etc. It should properly refer to the sources, which must be searchable, like 'In the original painting of the Mona Lisa, ...'
* The questions would be only 1 sentence long and the answer would be straight to the point. The question and answer both would be short. (not more than 40 words, ideally around 10-20 words)
* You will generate only the question based on these instructions and its answer in the format -
Question: 'Your multimodal question'
Answer: 'The answer for the multimodal question'
* Your response must include the keywords 'Question: ' and 'Answer: '

# Example
<user_query>
<img1 = A beige coloured curvy building with red sky in the background>
Caption for image source 1: National Museum of the American Indian in Washington, D.C
<img2 = A black and white photo of a building, most likely beige coloured, with a pond and some trees surrounding it>
Caption for image source 2: Xanadu-House-in-Kissimmee-Florida-1985 A photo of the Xanadu House that was located in Kissimmee, Florida, showing the exterior of the house.
Text Source: Construction of the Xanadu house in Kissimmee, Florida, began with the pouring of a concrete slab base and the erection of a tension ring 40 feet (12 m) in diameter to anchor the domed roof of what would become the \"Great Room\" of the house. (from: Xanadu Houses)
</user_query>

<model_thinking> (not in output)
Source Aspects: 
    Image Source 1:
        1. The building has a modern but curvy architecture, with bricks on the outside.
        2. The building is beige in color.
        3. The building appears to be a 3 storeyed one with an entrance on the ground floor.
    Image Source 2:
        1. The house looks alien-like with curved domes and circular windows.
        2. The image is black and white, however the house would be beige if colored.
        3. There are a collection of dome-like shapes constituting the house.
    Text Source:
        1. Xanadu house is in Kissimmee, Florida.
        2. The concrete slab base and the 40 feet diameter tension ring was constructed first to build the Xanadu house.
        3. The Great room of Xanadu house has a domed roof.
    Common Aspect: 'Circular/Curves in architecture', based on Image Source 1 - Point 1; Image Source 2 - Points 1,3; Text Source - Point 3.
</model_thinking>

<generator_response>
Question: Does the National Museum of the American Indian in Washington, D.C and the great room of Xanadu house have sharp corners in its architecture?
Answer: No they both have a curvy and rounded architecture.
</generator_response>
'''

# --- run if invoked ---
if __name__ == '__main__':
    input_path = INPUT_PATH_DEFAULT
    output_path = OUTPUT_BATCH_DEFAULT
    OUTPUT_DYNAMIC_DATASET = r"D:\MSc AI\Sem 2\Dissertation\Code v2\WebQA_dynamic_dataset_ITT_diff2.json"
    model = "gpt-4.1-mini-2025-04-14"
    n = 200
    random_sample = True
    seed = 42
    mode = "ITT_diff"   # change to "ITT_same", "ITT_diff", "IIT" as needed
    create_batch_jsonl(input_path, output_path, system_prompt, model, n, random_sample, seed, mode)
