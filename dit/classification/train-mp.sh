python -m torch.distributed.launch --nproc_per_node=2 --master_port=47770  run_class_finetuning.py \
        --model beit_base_patch16_224 \
        --data_path "/home/greg/datasets/dataset/rvlcdip" \
        --eval_data_path "/home/greg/datasets/dataset/rvlcdip" \
        --nb_classes 16 \
        --eval \
        --data_set rvlcdip \
        --output_dir output_dir \
        --log_dir output_dir/tf \
        --batch_size 16 \
        --abs_pos_emb \
        --disable_rel_pos_bias