import dotenv
from datasets import load_from_disk, concatenate_datasets, Dataset
import pandas as pd
from tqdm import tqdm
import random
from llama3.utils import get_openai_response
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
import pandas as pd
import os
import random

dotenv.load_dotenv()

def extract_answer(example):
  start_pos = example['cot_answer'].find("<ANSWER>") + len("<ANSWER>")
  example['answer'] = example['cot_answer'][start_pos:].strip()
  return example

def create_usr_msg(ds, num_questions):
    """Creates a user message with a random combination of questions and answers."""
    selected_indices = random.sample(range(len(ds)), num_questions)
    usr_msg = ""
    for i, idx in enumerate(selected_indices, 1):
        try:
            usr_msg += f"Question {i}: {ds[idx]['question']}\n"
            usr_msg += f"Answer {i}: {ds[idx]['answer']}\n"
            usr_msg += f"Context {i}: {ds[idx]['oracle_context']}\n"
        except IndexError:
            print(f"Warning: Not enough data for index {idx}. Skipping...")
    return usr_msg

def merge_batches(response_folder):
    # Merge all batches into a final dataset (including existing ones)
    merged_dataset = Dataset.from_pandas(
        pd.DataFrame(columns=["user_message", "response"])
    )
    for batch_file in os.listdir(response_folder):
        if batch_file.startswith("responses_batch_"):
            batch_path = os.path.join(response_folder, batch_file)
            batch_dataset = Dataset.load_from_disk(batch_path)
            merged_dataset = concatenate_datasets([merged_dataset, batch_dataset])
    return merged_dataset

def get_multi_qns(sys_msg, ds, num_res=100, batch_size=5, response_folder="response_batches"):
    # Check existing response batches
    existing_batches = [f for f in os.listdir(response_folder) if f.startswith("responses_batch_")]
    num_existing_responses = len(existing_batches) * batch_size
    num_new_res = num_res - num_existing_responses

    if num_new_res <= 0:
        return merge_batches(response_folder)

    responses = []
    response_data = []

    if not os.path.exists(response_folder):
        os.makedirs(response_folder)

    print(f"Generating {num_new_res} new responses...")

    for i in tqdm(range(num_new_res)):
        num_questions = random.choice([2, 3])  # Choose 2 or 3 questions randomly
        usr_msg = create_usr_msg(ds, num_questions)
        response = get_openai_response(sys_msg=sys_msg, usr_msg=usr_msg)

        if response is not None:
            responses.append(response)
            response_data.append({"user_message": usr_msg, "response": response})

        # Save in batches of 5 or when the loop ends
        if (i + 1) % batch_size == 0 or i == num_new_res - 1:
            response_dataset = Dataset.from_pandas(pd.DataFrame(response_data))
            response_dataset.save_to_disk(
                os.path.join(
                    response_folder,
                    f"responses_batch_{i // batch_size + len(existing_batches) + 1}",
                )
            )  # Adjust batch number based on existing batches
            response_data = []

    return merge_batches(response_folder)

if(__name__ == "__main__"):
    ds_rice = load_from_disk("./data/raft_qn")
    ds_wheat = load_from_disk("./data/raft_qn/wheat")
    ds = concatenate_datasets([ds_rice, ds_wheat]).shuffle() # type: ignore
    ds = ds.map(extract_answer)
    with open("./data/prompt_templates/multi_hop_qn_gen_sys.md", "r") as f:
        sys_msg = f.read()
    qns_ds = get_multi_qns(sys_msg, ds)
    qns_ds.save_to_disk("mhop_qns")
    print("saved generated dataset mhop_qns, you want to remove response_batch_* dirs we used for caching")


