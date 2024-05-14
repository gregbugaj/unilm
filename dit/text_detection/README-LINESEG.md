
# Convert the dataset

```bash
 python ./coco_funsd_dit_converter.py --coco_annoations_file ~/datasets/private/dit-linesegment/annotations/instances_default.json --output_file ~/datasets/private/dit-linesegment/converted/converted.json
```


Splitting the dataset
=====================
We split the dataset into training and test sets using the following command:

```bash
python ./cocosplit.py  ~/datasets/private/dit-linesegment/converted/converted.json ~/datasets/private/dit-linesegment/converted/instances_training.json ~/datasets/private/dit-linesegment/converted/instances_test.json -s .8
```


Visualizing dataset
==================

```shell
 python -m detectron2.data.datasets.coco ~/datasets/private/dit-linesegment/converted/instances_training.json ~/datasets/private/dit-linesegment/converted/imgs  dit_dataset 
``````


Training 
=========

For details see `README.md` ensure that the `data` folder is a symlink to the `~/datasets/private/dit-linesegment/converted/` folder or specific folder for the dataset.

```shell
ln -s ~/datasets/private/dit-linesegment/converted/ data
```
 

```shell
python train_net.py --config-file configs/mask_rcnn_dit_large.yaml --num-gpus 1 --resume MODEL.WEIGHTS ~/dev/marieai/marie-ai/model_zoo/unilm/dit/text_detection/tuned-2500-LARGE-v5-fixed/model_0104999.pth  OUTPUT_DIR ~/tmp/models/dit_linesegment_detection/tuned-01 SOLVER.IMS_PER_BATCH 1
```


Inference
=========

```shell
python ./inference.py --config-file configs/mask_rcnn_dit_large.yaml  --image_path ~/datasets/private/eob-extract/converted/imgs/eob-extract/eob-001  --output_path /tmp/dit-linesegment --opts  MODEL.WEIGHTS ~/tmp/models/dit_linesegment_detection/tuned-01/model_0000999.pth
```


