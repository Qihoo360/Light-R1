#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# adapted from https://github.com/huggingface/open-r1/pull/416

"""
This script is used to decontaminate a dataset by checking for n-gram overlap with other datasets.
It uses the same approach presented in https://arxiv.org/abs/2501.19393,
as found in: https://github.com/simplescaling/s1/blob/main/data/decontaminate_util.py
python scripts/decontaminate.py \
    --dataset "open-r1/verifiable-coding-problems-python" \
    --split train \
    --ngram_size 32 \
    --problem_column problem \
"""

import collections
from collections import defaultdict
from tqdm import tqdm
import json


def normalize_string(text: str) -> str:
    """Basic string normalization."""
    # Convert to lowercase and normalize whitespace
    text = text.lower().strip()
    # Replace multiple spaces with single space
    text = " ".join(text.split())
    return text


def word_ngrams(text: str, n: int) -> list:
    """Generate word-level n-grams from text."""
    words = text.split()
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def build_ngram_lookup(documents: list[str], ngram_size: int = 8) -> dict[str, set[int]]:
    """Build ngram lookup for documents."""
    lookup = collections.defaultdict(set)

    for doc_id, document in enumerate(tqdm(documents)):
        normalized_text = normalize_string(document)
        ngrams = word_ngrams(normalized_text, ngram_size)
        for ngram in ngrams:
            lookup[ngram].add(document)

    return lookup


def build_ngram_single(document: str, ngram_size: int = 8) -> set[str]:
    normalized_text = normalize_string(document)
    ngrams = word_ngrams(normalized_text, ngram_size)

    return set(ngrams)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to check for contamination.")
    parser.add_argument("--split", type=str, default="train", help="Split to check for contamination, defaults to `train`.")
    parser.add_argument("--ngram_size", type=int, default=32, help="Size of n-grams to build, defaults to 8.")
    parser.add_argument("--problem_column", type=str, default="problem", help="Name of the column containing the problem (prompt).")
    parser.add_argument("--format", type=str, default='Dataset', help="Input format, supports `json` and `Dataset`.")
    args = parser.parse_args()

    from datasets import load_dataset, Dataset

    # Load the dataset to check for contamination
    if args.format == 'json':
        ds = load_dataset("json", data_files=args.dataset, split=args.split)
    else:
        ds = load_dataset(args.dataset, split=args.split)

    eval_datasets = {
        "aime_2024": (load_dataset("HuggingFaceH4/aime_2024", split="train"), "problem"),
        "aime_2025": (load_dataset("yentinglin/aime_2025", split="train"), "problem"),
        "math_500": (load_dataset("HuggingFaceH4/MATH-500", split="test"), "problem"),
        "gpqa": (load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", trust_remote_code=True), "Question"),
    }
    # example to decontaminate against the above 4 datasets. Internally we decontaminated against more.

    ngram_lookups = {}
    print('-----build ngram lookup------')
    for ds_name, (eval_dataset, problem_col) in eval_datasets.items():
        ngram_lookups[ds_name] = build_ngram_lookup(eval_dataset[problem_col], ngram_size=args.ngram_size)

    filtered_samples = [] 
    contamination_counts = defaultdict(int)

    def find_contaminated(row):
        # For each example we have to build the ngrams and check for all of them on each row
        if args.problem_column == 'conversations':
            ngrams = build_ngram_single(row[args.problem_column][0]['value'], ngram_size=args.ngram_size)
        else:
            ngrams = build_ngram_single(row[args.problem_column], ngram_size=args.ngram_size)
        
        is_contaminated = any(ngram in ngram_lookup for ngram in ngrams)
        matched_ngrams = [ngram for ngram in ngrams if ngram in ngram_lookup] if is_contaminated else []
        
        row[f"contaminated_{eval_name}"] = is_contaminated
        if is_contaminated:
            if args.problem_column == 'conversations' or args.problem_column == 'messages':
                sample = {
                    "filtered_prompt": row[args.problem_column][0]['value'],
                    "eval_dataset": eval_name,
                    "matched_ngrams": matched_ngrams,
                    "eval_dataset_entries": list(ngram_lookup[matched_ngrams[0]]) if matched_ngrams else []
                }
            else:
                sample = {
                    "filtered_prompt": row[args.problem_column],
                    "eval_dataset": eval_name,
                    "matched_ngrams": matched_ngrams,
                    "eval_dataset_entries": list(ngram_lookup[matched_ngrams[0]]) if matched_ngrams else []
                }
            print(sample)
        
        return row

    for eval_name, ngram_lookup in ngram_lookups.items():
        results = ds.map(find_contaminated, num_proc=64)

        for row in results:
            if row[f"contaminated_{eval_name}"]:
                contamination_counts[eval_name] += 1

    print("----------Contaminated Data Statistics----------")
    if not contamination_counts:
        print("No data contamination")
    else:
        for eval_name, count in contamination_counts.items():
            print(f"Dataset: {eval_name}, Contaminated Data Number: {count}")
