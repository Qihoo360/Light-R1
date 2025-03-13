import pandas as pd
import json
import re
from datasets import load_dataset, Dataset, load_from_disk
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to check for contamination.")
parser.add_argument("--split", type=str, default="train", help="Split to check for contamination, defaults to `train`.")
parser.add_argument("--problem_column", type=str, default="problem", help="Name of the column containing the problem (prompt). Problem_column supports conversations, messages and others.")
parser.add_argument("--format", type=str, default="Dataset", help="Input format, supports `json` and `Dataset`.")
parser.add_argument("--output_json", action="store_true", default=False, help="Whether to write polluted data to a JSON file. Defaults to False.")

args = parser.parse_args()
train_data_f = args.dataset
input_format = args.format

def clean_text(text):
    text = re.sub(r'[^\w]', '', text) 
    text = re.sub(r'\d+', '', text)   
    text = text.lower()    
    return text

eval_datasets = {
    "aime_2024": (load_dataset("HuggingFaceH4/aime_2024", split="train"), "problem"),
    "aime_2025": (load_dataset("yentinglin/aime_2025", split="train"), "problem"),
    "math_500": (load_dataset("HuggingFaceH4/MATH-500", split="test"), "problem"),
    "gpqa": (load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", trust_remote_code=True), "Question"),
}
# example to decontaminate against the above 4 datasets. Internally we decontaminated against more.

# load benchmark datas
benchmark_prompts_dct = dict()

for dataset_name, (dataset, prompt_key) in eval_datasets.items():
    print(f"dataset:{dataset_name}, prompt_key:{prompt_key}")
    for item in dataset:
        item["source"] = dataset_name
        prompt = item[prompt_key]
        clean_prompt = clean_text(prompt)
        benchmark_prompts_dct[prompt] = item
        benchmark_prompts_dct[clean_prompt] = item

# get training data
training_prompts = list()
prompt_mode = 0
if args.format == 'json':  # problem_column supports conversations, messages and others
    if args.problem_column == 'conversations' or args.problem_column == 'messages':
        try:
            with open(train_data_f, 'r') as f:
                jsons = json.load(f)
            training_prompts = [j["conversations"][0]["value"] for j in jsons]
            prompt_mode = 1
        except:
            try:
                with open(train_data_f, 'r') as f:
                    jsons = json.load(f)
                training_prompts = [j["messages"][0]["content"] for j in jsons]
                prompt_mode = 2
            except:
                try:
                    with open(train_data_f, 'r') as f:
                        lines = f.readlines()
                    jsons = [json.loads(l) for l in lines]
                    training_prompts = [j["messages"][0]["content"] for j in jsons]
                    prompt_mode = 3
                except:
                    raise NotImplementedError("Implement your own way to get the list of training prompts here!")
    else:
        with open(train_data_f, 'r') as f:
            jsons = json.load(f)
        training_prompts = [j[args.problem_column] for j in jsons]
        prompt_mode = 4
    print(len(training_prompts))
    assert len(training_prompts) > 0 and prompt_mode > 0
elif args.format == 'Dataset':
    try:
        ds = load_dataset(args.dataset, split=args.split)
    except:
        try:
            ds = load_from_disk(args.dataset)
        except:
            raise NotImplementedError("Implement your own way to get the list of training prompts here!")
    if args.problem_column == 'conversations' or args.problem_column == 'messages':
        try:
            training_prompts = [j["conversations"][0]["value"] for j in ds]
            prompt_mode = 1
        except:
            try:
                training_prompts = [j["messages"][0]["content"] for j in ds]
                prompt_mode = 2
            except:
                try:
                    training_prompts = [json.loads(j)["messages"][0]["content"] for j in ds]
                    prompt_mode = 3
                except:
                    raise NotImplementedError("Implement your own way to get the list of training prompts here!")
    else:
        training_prompts = [j[args.problem_column] for j in ds]
        prompt_mode = 4
    print(len(training_prompts))
    assert len(training_prompts) > 0 and prompt_mode > 0

contaminated_stats = {}
contaminated_lst = list()
cleaned_lst = list()

if args.format == 'json':
    for j in jsons:
        if prompt_mode == 1:
            train_prompt = j["conversations"][0]["value"]
        elif prompt_mode == 2:
            train_prompt = j["messages"][0]["content"]
        elif prompt_mode == 3:
            train_prompt = j["messages"][0]["content"]
        elif prompt_mode == 4:
            train_prompt = j[args.problem_column]
        clean_prompt = clean_text(train_prompt)
        if train_prompt in benchmark_prompts_dct or clean_prompt in benchmark_prompts_dct:
            print("There is data pollution!")
            if train_prompt in benchmark_prompts_dct:
                detail_info = benchmark_prompts_dct[train_prompt]
            elif clean_prompt in benchmark_prompts_dct:
                detail_info = benchmark_prompts_dct[clean_prompt]
            
            contaminated_entry = {
                "train_prompt": train_prompt,
                "benchmark_prompt": detail_info,
                "source": detail_info.get('source', 'unknown')
            }
            contaminated_lst.append(contaminated_entry)
            source = detail_info.get('source', 'unknown')
            if source in contaminated_stats:
                contaminated_stats[source] += 1
            else:
                contaminated_stats[source] = 1
        else:
            cleaned_lst.append(j)
elif args.format == 'Dataset':
    for j in ds:
        if prompt_mode == 1:
            train_prompt = j["conversations"][0]["value"]
        elif prompt_mode == 2:
            train_prompt = j["messages"][0]["content"]
        elif prompt_mode == 3:
            train_prompt = j["messages"][0]["content"]
        elif prompt_mode == 4:
            train_prompt = j[args.problem_column]
        clean_prompt = clean_text(train_prompt)
        if train_prompt in benchmark_prompts_dct or clean_prompt in benchmark_prompts_dct:
            print("There is data pollution!")
            if train_prompt in benchmark_prompts_dct:
                detail_info = benchmark_prompts_dct[train_prompt]
            elif clean_prompt in benchmark_prompts_dct:
                detail_info = benchmark_prompts_dct[clean_prompt]
            
            contaminated_entry = {
                "train_prompt": train_prompt,
                "benchmark_prompt": detail_info,
                "source": detail_info.get('source', 'unknown')
            }
            contaminated_lst.append(contaminated_entry)
            source = detail_info.get('source', 'unknown')

            if source in contaminated_stats:
                contaminated_stats[source] += 1
            else:
                contaminated_stats[source] = 1
        else:
            cleaned_lst.append(j)

print("----------Contaminated Data Statistics----------")
if not contaminated_stats:
    print("No data contamination")
else:
    for source, count in contaminated_stats.items():
        print(f"Source: {source}, Contaminated Data Number: {count}")

if args.output_json:
    import os
    base_output_name = os.path.basename(train_data_f)
    output_file = f"{base_output_name}_contaminated_entries.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(contaminated_lst, f, indent=4, ensure_ascii=False)
    print(f"Pollution data entries saved to {output_file}")