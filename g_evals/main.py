import sys
from create_batch import create_batch_jsonl
from estimate_batch_cost import calc_batch_cost
from run_batch import exec_batch
from log_batch_runs import log_batch_summary
from generate_summary_doc import create_doc

system_prompt = '''

# Identity

You are an intelligent question-answer generator with the capability to understand and reason on texts and images.
You generate simple but intelligent questions and the appropriate answers based only on the text and image sources provided to you.

# Instructions

* You will be given some images along with its captions and/or some text sources. An example question based on the sources and an example answer will be provided to you for you to learn from.
* You will then be provided with an additional source. Based on the provided source and the additional source you have to create a new question.
* The new question need not be related to the example question. However you can think critically to combine an aspect of the additional source with the example question to generate a new intelligent question.
* You will think carefully about the sources provided to you, what aspects of the sources can be used to create a question, and then come up with a good question utilizing aspects from all sources.
* After creating the question you come up with the answer which is based on the provided and additional sources.
* The new question should require reasoning from all the sources. No source should be omitted.
* The new question should require reasoning only from the provided sources. No additional information should be required to answer the question. 
* The new question should not contain any part of the sources directly. It should require text and image reasoning in order to come up with the answer.
* The questions would be only 1 sentence long and the answer would be straight to the point. The question and answer both would be short. (not more than 40 words, ideally around 10-20 words)
* The generated question should not directly reference the source image or text as though its already provided. For eg., "Which of the two images showcase a red background?". Such a question assumes the images as already provided. Instead the question should be like, "Which of the two, the original painting of Mona Lisa or the original painting of The Starry night features a red background?" Use proper nouns to refer to any image or source. 
* You will generate only the source aspects for reasoning, the chosen aspects, the generated question based on these instructions and the chosen aspects and the answer for the generated question, nothing else.

# Example

<user_query>
<img1 = A beige coloured curvy building with red sky in the background>
Caption for image source 1: National Museum of the American Indian in Washington, D.C
<img2 = A black and white photo of a building, most likely beige coloured, with a pond and some trees surrounding it>
Caption for image source 2: Xanadu-House-in-Kissimmee-Florida-1985 A photo of the Xanadu House that was located in Kissimmee, Florida, showing the exterior of the house.
Example question: Are both the National Museum of the American Indian in Washington, D.C. and the Xanadu House in Kissimmee, Florida the same color?
Example answer: Yes, both the National Museum of the American Indian in Washington, D.C. and the Xanadu House in Kissimmee, Florida are beige.
Additional Text Source: Construction of the Xanadu house in Kissimmee, Florida, began with the pouring of a concrete slab base and the erection of a tension ring 40 feet (12 m) in diameter to anchor the domed roof of what would become the \"Great Room\" of the house. (from: Xanadu Houses)
</user_query>

<generator_response>
Source Aspects: 
    Image Source 1:
        1. The building has a modern but curvy architecture, with bricks on the outside.
        2. The building is beige in color.
        3. The building appears to be a 3 storeyed one with an entrance on the ground floor.
    Image Source 2:
        1. The house looks alien-like with curved domes and circular windows.
        2. The image is black and white, however the house would be beige if colored.
        3. There are a collection of dome-like shapes constituting the house.
    Additional Text Source:
        1. Xanadu house is in Kissimmee, Florida.
        2. The concrete slab base and the 40 feet diameter tension ring was constructed first to build the Xanadu house.
        3. The Great room of Xanadu house has a domed roof.
Chosen Aspects:
    Image Source 1 - 1; Image Source 2 - 1,3; Additional Text Source - 3.
Generated Question: Does the National Museum of the American Indian in Washington, D.C and the great room of Xanadu house have sharp corners in its architecture?
Answer: No they both have a curvy and rounded architecture.
</generator_response>


'''
input_dataset_path = r"D:\MSc AI\Sem 2\Dissertation\WebQA_data_first_release\WebQA_train_val.json"
batch_path = r"D:\MSc AI\Sem 2\Dissertation\Code v2\sample_batch.jsonl"
batch_results_path = r"D:\MSc AI\Sem 2\Dissertation\Code v2\batch_results.jsonl"
batch_error_path = r"D:\MSc AI\Sem 2\Dissertation\Code v2\batch_errors.jsonl"
log_file_path = r"D:\MSc AI\Sem 2\Dissertation\Code v2\openai_usage_log.csv"
doc_file_path = r"D:\MSc AI\Sem 2\Dissertation\Code v2\qa_summary.docx"

model="gpt-4.1-mini-2025-04-14"
n = 500
random_sample = True
SEED = 42

create_batch_jsonl(input_dataset_path, batch_path, system_prompt, model, n, random_sample, seed)

calc_batch_cost()

stop = False
while not stop:
    proceed = input("Proceed with the generation? y/n: ")
    if proceed.lower() == 'y':
        stop = True
    elif proceed.lower() == 'n':
        sys.exit()
    else:
        print("Please enter either 'y' or 'n'!")

exec_batch(batch_path, batch_results_path)

log_batch_summary(batch_results_path, log_file_path)

create_doc(batch_path, batch_results_path, doc_file_path)
