import os
import csv
import subprocess
import shutil
import pandas as pd
import requests
import time
import signal
import json
import numpy as np
import argparse

from tabulate import tabulate
from collections import Counter
from verl.utils.hdfs_io import makedirs
from verl.utils.fs import copy_local_path_from_hdfs
from deepscaler.rewards.math_reward import deepscaler_reward_fn
from deepscaler.rewards.math_utils.utils import extract_answer


def find_mode(lst):
    if len(lst) == 0:
        return list()
    counter = Counter(lst)
    max_count = max(counter.values())
    mode = [k for k, v in counter.items() if v == max_count]
    return mode


# Add the select_reward_fn from main_eval.py
def select_reward_fn(data_source):
    if data_source == 'lighteval/MATH':
        from verl.utils.reward_score import math
        return math.compute_score
    else:
        from deepscaler.rewards.math_reward import deepscaler_reward_fn
        return deepscaler_reward_fn


def compute_correctness(dataset):
    total_lst = list()
    for i in range(len(dataset)):
        row = dataset.iloc[i]
        prompt = row['prompt']
        gt = row['reward_model']['ground_truth']
        # print(gt)
        responses_this = row['responses']

        true_false = [int(deepscaler_reward_fn(response, gt, skip_format_reward=True)) for response in responses_this]
        total_lst.append(true_false)
    return total_lst

def light_r1_postprocessing(dataset, model_path, save_path, benchmark):
    start_time = time.time()
    os.environ['NCCL_DEBUG'] = 'WARN'
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    local_path = copy_local_path_from_hdfs(model_path)
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    ###config hard-coded###
    rollout_response_length = 16384
    skip_format_reward = True
    n_samples = 64

    ###Light R1 postprocessing###
    # add correctness field
    total_lst = compute_correctness(dataset)
    dataset['correctness'] = total_lst

    if 'correctness' not in dataset:
        total_lst = compute_correctness(dataset)
        dataset['correctness'] = total_lst
        dataset.to_json(f'{save_path}/{benchmark}.json', orient='records', force_ascii=False, lines=True)
    
    # Compute evaluation metrics
    prompts = dataset['prompt']
    responses = dataset['responses']  # Using the generated responses
    data_sources = dataset['data_source']
    reward_model_data = dataset['reward_model']        

    output_lst = [str(r) for responses_this in list(responses) for r in responses_this]
    # print(output_lst)
    print(type(output_lst), type(output_lst[0]))
    unpad_tokenized = tokenizer(output_lst, add_special_tokens=False).input_ids
    len_response_tokens = [len(tokens) for tokens in unpad_tokenized]
    len_mean = np.mean(len_response_tokens)
    cutoff_ratio = sum([l == rollout_response_length for l in len_response_tokens]) / len(unpad_tokenized)
    print('length cutoff ratio:', cutoff_ratio)

    passes = 0    
    total = len(dataset)
    total_scores = []
    total_num_responses = 0  # 전체 response 수
    total_score_sum = 0  # 모든 response의 score의 총합

    conses = 0
    
    for i in range(total):
        response_lst = responses[i]
        data_source = data_sources[i]
        prompt = prompts[i]
        reward_data = reward_model_data[i]
        reward_fn = select_reward_fn(data_source)
        ground_truth = reward_data['ground_truth']
        score_lst = []

        for r in response_lst:
            try:
                if skip_format_reward:
                    score = reward_fn(r, ground_truth, skip_format_reward=True)
                else:
                    score = reward_fn(r, ground_truth, skip_format_reward=False)
            except:  # 没字段表示没指定该参数，默认跳过格式校验
                score = reward_fn(r, ground_truth, skip_format_reward=True)
            score_lst.append(score)

        total_scores.append(score_lst)
        total_score_sum += sum(score_lst)
        total_num_responses += len(score_lst)

        max_score = np.max(score_lst)
        if max_score == 1:
            passes += 1
        
        extracted_lst = [extract_answer(r) for r in response_lst]
        extracted_lst = [r for r in extracted_lst if r is not None]
        cons_answers = find_mode(extracted_lst)
        cons_response_lst = [r for r in response_lst if extract_answer(r) in cons_answers]
        is_cons_correct_list = list()
        for r in cons_response_lst:
            try:
                if skip_format_reward:
                    score = reward_fn(r, ground_truth, skip_format_reward=True)
                else:
                    score = reward_fn(r, ground_truth, skip_format_reward=False)
            except:  # 没字段表示没指定该参数，默认跳过格式校验
                score = reward_fn(r, ground_truth, skip_format_reward=True)
            is_cons_correct_list.append(score)
        if any(is_cons_correct_list):
            conses += np.mean(is_cons_correct_list)

    # 변경된 pass@1 방식 (micro-avg)
    pass_at_1 = np.mean(total_scores)
    pass_at_new_1 = total_score_sum / total_num_responses
    pass_at_n = passes / total
    cons_at_n = conses / total

    spent_time = time.time() - start_time
    spent_hours = spent_time / 60 / 60
    # Save metrics to CSV
    # csv_path = os.path.join(output_dir, f'pass_{spent_hours:.2f}h.csv')
    output_dir = os.path.dirname(f'{save_path}/{benchmark}.parquet')
    makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f'{benchmark}_overall_results.csv')

    # Prepare the row data
    # Extract the dataset name from the path
    row_data = {
        'model_path': model_path,
        'dataset': benchmark,
        # 'pass@1': pass_at_1,
        f'pass@1': pass_at_new_1,
        f'pass@{n_samples}': pass_at_n,
        f'cons@{n_samples}': cons_at_n,
        'cutoff_raio': cutoff_ratio,
        'mean_response_tokens': len_mean,
        'run_hours': spent_hours
    }

    # Check if file exists
    file_exists = os.path.isfile(csv_path)
    
    # Write to CSV
    with open(csv_path, mode='a', newline='') as f:  # 追加写，不会覆盖，所以没事
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    # Convert the row data into a list of lists format for tabulate
    table_data = [[k, v] for k, v in row_data.items()]
    
    # Print table
    print(tabulate(table_data, headers=['Metric', 'Value'], tablefmt='grid'))

    ###Light R1 postprocessing Done###

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='light-r1-postprocess')
    parser.add_argument('--model_path', type=str, required=True, help='model path')
    parser.add_argument('--save_path', type=str, required=True, help='output directory')
    parser.add_argument('--benchmark', type=str, required=True, help='benchmark')
    args = parser.parse_args()
    dataset = pd.read_parquet(args.save_path + "/" + args.benchmark + ".parquet")
    light_r1_postprocessing(dataset, args.model_path, args.save_path, args.benchmark)
