
https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/trainer#transformers.TrainingArguments.per_device_train_batch_size

```
CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$PWD" python -m torch.distributed.launch --nproc_per_node=2 run_funsd.py \
        --model_name_or_path microsoft/layoutlm-base-uncased \
        --output_dir /home/gbugaj/dev/unilm/layoutlmft/examples/checkpoints \
        --do_train \
        --do_predict \
        --num_train_epochs 100 \
        --per_device_train_batch_size 16 \
        --warmup_ratio 0.1 \
        --deepspeed ds_config_1gpu.json \
        --fp16 
```


Add DeepSpeed support 
https://huggingface.co/docs/transformers/main_classes/deepspeed