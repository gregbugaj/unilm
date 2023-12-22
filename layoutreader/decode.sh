#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python decode_seq2seq.py --fp16 \
    --model_type layoutlm \
    --tokenizer_name bert-base-uncased \
    --input_folder /path/to/ReadingBank/test \
    --cached_feature_file /path/to/ReadingBank/features_test.pt \
    --output_file /path/to/output/LayoutReader/layoutlm/output.txt \
    --split test \
    --do_lower_case \
    --model_path /path/to/output/LayoutReader/layoutlm/ckpt-75000 \
    --cache_dir /path/to/output/LayoutReader/cache \
    --max_seq_length 1024 \
    --max_tgt_length 511 \
    --batch_size 32 \
    --beam_size 1 \
    --length_penalty 0 \
    --forbid_duplicate_ngrams \
    --mode s2s \
    --forbid_ignore_word "."