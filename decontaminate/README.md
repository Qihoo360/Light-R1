# Data Decontamination: N-gram and Exact Matching

We employ two methods to remove data pollution: **n-gram-based matching** and **character-based exact matching**.

## N-gram Matching
We adapt the n-gram matching rules from the [open-r1](https://github.com/huggingface/open-r1/pull/416) script with some modifications. To prevent the removal of locally similar prompts. Specifically, we used **32-gram** for deduplication.

To run the script, download the [datasets](https://huggingface.co/datasets/qihoo360/Light-R1-SFTData) and execute the following command:

```shell
python ./decontaminate/n_gram_check.py --dataset ./stage2-3k.json --split train --problem_column conversations --ngram_size 32 --format json
```

Parameters:
- **dataset** (`str`): Path to the dataset file.
- **split** (`str`): Dataset split to process, defaults to `"train"`.
- **problem_column** (`str`): Field containing the prompt. For `"conversations"`, it extracts `"conversations[value][0]"`.
- **ngram_size** (`int`): Size of n-grams for deduplication.
- **format** (`str`): Input format, supports `"json"` and `"Dataset"`.

## Exact Matching
To address data contamination caused by prompts with only numerical differences, we add an **exact matching** step. Numeric parts of the prompts are removed before matching.

Run the following script to perform exact matching:

```shell
python ./decontaminate/character_matching.py --dataset ./stage2-3k.json --split train --format json --problem_column conversations
```

Parameters:
- **dataset** (`str`): Path to the dataset file.
- **split** (`str`): Dataset split to process, defaults to `"train"`.
- **problem_column** (`str`): Field containing the prompt. For `"conversations"`, it extracts `"conversations[value][0]"`.
- **format** (`str`): Input format, supports `"json"` and `"Dataset"`.
- **output_json** (`bool`, default=`False`, action=`False`): Whether to write polluted data to a JSON file. Defaults to False.


You can also run the following script to perform both n-gram and exact matching:

```shell
./decontaminate/check_data.sh
```