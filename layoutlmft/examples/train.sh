#!/bin/bash

export BS=16
export MAX_JOBS=8
# CUDA_VISIBLE_DEVICES=1 

# --max_train_samples 5000 \
# --max_test_samples 1000 \
# --model_name_or_path microsoft/layoutlmv2-large-uncased \
# --fp16 
# --model_name_or_path /home/gbugaj/dev/unilm/layoutlmft/examples/checkpoints/checkpoint-9500/ \
# --resume_from_checkpoint /home/gbugaj/dev/unilm/layoutlmft/examples/checkpoints/checkpoint-8500 \
# --resume_from_checkpoint /home/gbugaj/dev/unilm/layoutlmft/examples/checkpoints-tuned-pan/checkpoint-250 \
# --deepspeed ds_config_gpu.json \

# --greater_is_better false \
# --metric_for_best_model loss \

PYTHONPATH="$PWD" python -m torch.distributed.launch --nproc_per_node=2 run_funsd.py \
        --model_name_or_path microsoft/layoutlmv2-large-uncased \
        --output_dir /mnt/data/models/layoutlmv2-large-finetuned-funsd\
        --overwrite_output_dir \
        --save_steps 250 \
        --save_total_limit 10 \
        --save_strategy steps \
        --load_best_model_at_end true \
        --evaluation_strategy steps \
        --eval_steps 250 \
        --report_to tensorboard \
        --logging_dir  ./logs \
        --log_level debug \
        --logging_strategy steps \
        --logging_steps 50 \
        --do_train \
        --do_eval \
        --num_train_epochs 50 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 4 \
        --warmup_ratio 0.1 \
        --warmup_steps 10 \
        --learning_rate 1e-5
        --fp16
