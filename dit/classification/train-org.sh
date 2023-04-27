exp_name=dit-base-exp
mkdir -p output/${exp_name}

# Download the model
if [ ! -f "/home/greg/dev/models/dit/dit-base-224-p16-500k-62d53a.pth" ]; then
    wget https://dl.fbaipublicfiles.com/deit/dit_base_patch16_224-a1311bcf.pth -O /home/greg/dev/models/dit/dit-base-224-p16-500k-62d53a.pth
fi

python -m torch.distributed.launch --nproc_per_node=2 --master_port=47770  run_class_finetuning.py \
        --model deit_base_patch16_224 \
        --data_path "/home/greg/datasets/dataset/rvlcdip" \
        --eval_data_path "/home/greg/datasets/dataset/rvlcdip" \
        --enable_deepspeed \
        --nb_classes 16 \
        --data_set rvlcdip \
        --finetune "/home/greg/dev/models/dit/dit-base-224-p16-500k-62d53a.pth" \
        --output_dir output/${exp_name}/ \
        --log_dir output/${exp_name}/tf \
        --batch_size 16 \
        --lr 5e-4  \
        --update_freq 2 \
        --eval_freq 1 \
        --save_ckpt_freq 1 \
        --epochs 30  \
        --layer_scale_init_value 1e-5 \
        --layer_decay 0.75  \
        --drop_path 0.2  \
        --weight_decay 0.05 \
        --clip_grad 1.0 \
        --abs_pos_emb \
        --disable_rel_pos_bias


