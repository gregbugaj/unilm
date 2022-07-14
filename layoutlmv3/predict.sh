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
  

PYTHONPATH="$PWD" python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port 4500 examples/run_funsd.py \
    --dataset_name funsd \
    --do_eval \
    --model_name_or_path /data/models/layoutlmv3-base-finetuned/checkpoint-4000 \
    --output_dir /data/models/layoutlmv3-base-finetuned-eval \
    --segment_level_layout 1 --visual_embed 1 --input_size 224 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 8 \
    --fp16
