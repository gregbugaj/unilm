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

#      --pad_to_max_length true \
#     --label_all_tokens true \
  

# tensorboard --logdir=/data/models/layoutlmv3-base-finetuned --bind_all
#      
    # --pad_to_max_length true \
    # --label_all_tokens true \
    # --fp16
   

PYTHONPATH="$PWD" python -m torch.distributed.launch \
    --nproc_per_node=2 --master_port 4398 examples/run_funsd.py \
    --dataset_name funsd \
    --resume_from_checkpoint /home/greg/tmp/models/layoutlmv3-base-finetuned/checkpoint-500 \
    --do_train \
    --do_eval \
    --model_name_or_path microsoft/layoutlmv3-base \
    --output_dir /data/models/layoutlmv3-base-finetuned \
    --segment_level_layout 1 --visual_embed 1 --input_size 224 \
    --max_steps 10000 \
    --save_steps 500 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --return_entity_level_metrics true \
    --dataloader_num_workers 8 \
    --label_all_tokens true \
    --cache_dir /tmp/cache/ \
    --fp16
