export DATA=/home/greg/datasets/SROIE_FineTune

export DATA=/home/greg/datasets/SROIE_OCR/ready
export MODEL_BASE=/mnt/data/marie-ai/model_zoo/trocr/trocr-large-printed.pt
# export MODEL_BASE=/data/models/unilm/trocr/ft_SROIE_LINES_SET18/checkpoint_best.pt

export MODEL_NAME=ft_SROIE_LINES_SET41
export SAVE_PATH=/data/models/unilm/trocr/${MODEL_NAME}
export LOG_DIR=log_${MODEL_NAME}

mkdir ${LOG_DIR}
export BSZ=8
export valid_BSZ=6

$(which fairseq-train) \
    --data-type SROIE --user-dir ./ --task text_recognition --input-size 384 \
    --arch trocr_large \
    --seed 1234 --optimizer adam --lr 2e-05 --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-8 --warmup-updates 800 --weight-decay 0.0001 --log-format tqdm \
    --log-interval 10 --batch-size ${BSZ} --batch-size-valid ${valid_BSZ} \
    --save-dir ${SAVE_PATH} --tensorboard-logdir ${LOG_DIR} --max-epoch 5000 \
    --patience 16 --ddp-backend legacy_ddp --num-workers 10 --preprocess DA2 \
    --bpe gpt2 --decoder-pretrained roberta2 \
    --update-freq 16 --finetune-from-model ${MODEL_BASE} --fp16 \
    ${DATA}