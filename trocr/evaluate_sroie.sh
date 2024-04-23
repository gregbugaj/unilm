export DATA=/home/greg/datasets/SROIE_Task2_Original
export MODELXX=/mnt/data/marie-ai/model_zoo/trocr/trocr-large-printed.pt
export MODEL=/home/greg/models/unilm/trocr/ft_SROIE/checkpoint_best.pt
export RESULT_PATH=/tmp/result
export BSZ=16


$(which fairseq-generate) \
        --data-type SROIE --user-dir ./ --task text_recognition --input-size 384 \
        --beam 10 --nbest 1 --scoring sroie --gen-subset test \
        --batch-size ${BSZ} --path ${MODEL} --results-path ${RESULT_PATH} \
        --bpe gpt2 --dict-path-or-url ./gpt2_with_mask.dict.txt \
        --preprocess DA2 \
        --fp16 \
        ${DATA}