python character_matching.py --dataset ./stage2-3k.json --split train --format json --problem_column conversations

python n_gram_check.py --dataset ./stage2-3k.json --split train --problem_column conversations --ngram_size 32 --format json