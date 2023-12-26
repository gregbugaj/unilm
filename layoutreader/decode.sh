#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# --cached_feature_file /path/to/ReadingBank/features_test.pt \

python decode_seq2seq.py --fp16 \
    --model_type layoutlm \
    --tokenizer_name bert-base-uncased \
    --input_folder /mnt/data/marie-ai/model_zoo/unilm/layoutreader/ReadingBank/fast \
    --output_file /mnt/data/marie-ai/model_zoo/unilm/layoutreader/output.txt \
    --split test \
    --do_lower_case \
    --model_path /mnt/data/marie-ai/model_zoo/unilm/layoutreader/layoutreader-base-readingbank \
    --cache_dir /mnt/data/marie-ai/model_zoo/unilm/layoutreader/cache \
    --max_seq_length 1024 \
    --max_tgt_length 511 \
    --batch_size 32 \
    --beam_size 1 \
    --length_penalty 0 \
    --forbid_duplicate_ngrams \
    --mode s2s \
    --forbid_ignore_word "."