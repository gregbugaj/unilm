#!/bin/bash

export BS=16
export MAX_JOBS=8
# CUDA_VISIBLE_DEVICES=1 

# --max_train_samples 5000 \
# --max_test_samples 1000 \
# --model_name_or_path microsoft/layoutlmv2-large-uncased \
# --fp16 
# --model_name_or_path /home/gbugaj/dev/unilm/layoutlmft/examples/checkpoints/checkpoint-9500/ \

PYTHONPATH="$PWD" python -m torch.distributed.launch --nproc_per_node=2 run_funsd.py \
        --model_name_or_path microsoft/layoutlmv2-large-uncased \
        --output_dir /home/gbugaj/dev/unilm/layoutlmft/examples/checkpoints-tuned \
        --overwrite_output_dir \
        --resume_from_checkpoint true \
        --load_best_model_at_end true \
        --save_steps 500 \
        --save_strategy steps \
        --evaluation_strategy steps \
        --eval_steps 100 \
        --save_total_limit 10 \
        --report_to tensorboard \
        --logging_dir  ./logs \
        --log_level debug \
        --logging_strategy steps \
        --logging_steps 100\
        --do_train \
        --do_predict \
        --num_train_epochs 10 \
        --per_device_train_batch_size 8 \
        --warmup_ratio 0.1 \
        --warmup_steps 2 \
        --deepspeed ds_config_gpu.json \
        --tf32 true
