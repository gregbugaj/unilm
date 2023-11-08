#!/bin/bash

export BS=16
export MAX_JOBS=8
CUDA_VISIBLE_DEVICES=1 

# tensorboard --logdir=/data/models/layoutlmv3-base-finetuned --bind_all
#      
    # --pad_to_max_length true \
    # --label_all_tokens true \
    # --fp16
   # /data/models/layoutlmv3-base-finetuned 

# --resume_from_checkpoint /mnt/data/models/layoutlmv3-base-finetuned/checkpoint-1000 \
    # --resume_from_checkpoint /mnt/data/models/layoutlmv3-base-finetuned-segment_level_layout/checkpoint-500 \

#--resume_from_checkpoint /mnt/data/models/layoutlmv3-large-finetuned/checkpoint-1000 \

PYTHONPATH="$PWD" python examples/run_funsd.py \
    --dataset_name funsd \
    --do_train \
    --do_eval \
    --model_name_or_path microsoft/layoutlmv3-large \
    --output_dir /mnt/data/models/layoutlmv3-large-patpay \
    --segment_level_layout 0 \
    --visual_embed 1 \
    --input_size 224 \
    --max_steps 10000 \
    --save_steps 250 \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --logging_steps 10 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --return_entity_level_metrics false \
    --dataloader_num_workers 8 \
    --cache_dir /tmp/cache/ \
    --preprocessing_num_workers 1 \
    --overwrite_output_dir \
    --pad_to_max_length true \
    --label_all_tokens false \
    --fp16
