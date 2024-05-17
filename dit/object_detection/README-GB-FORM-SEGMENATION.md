# FORM SEGMENTATION / DOCUMENT BOUNDARY DETECTION


Data preparation
================


Data preparation consist of following steps:
* Merging multiple  datasets into a single dataset (Multiple datasets are merged into a single dataset to increase the size of the dataset. This is done to improve the performance of the model.)

* Convert the dataset to COCO format
* Split the dataset into training and test sets
* Link the dataset to the detectron2 dataset catalog
* Train the model

Root directory for the dataset is `~/datasets/private/form-segmenation` as per convertion the `converted` directory is used to store the converted datasets.

Shared tools are located in `~/dev/unilm/dit/tools`


Merge the annotation files from the different datasets.
=======================================================

We merge the annotations from the different datasets into a single one. 

```shell
python ./merge_dir.py --src_dir ~/datasets/private/form-segmenation/raw --output_file ~/datasets/private/form-segmenation/converted/merged.json
```

Converting the dataset.
=======================

The dataset is converted to the DIT format using the following command:

Depending on the dataset you might have to adjust the `category_id` in the `coco_funsd_dit_converter.py` script. This is the category id that will be used in the COCO annotations. 

```shell
python ./coco_funsd_dit_converter.py --coco_annoations_file ~/datasets/private/form-segmenation/converted/merged.json --output_file ~/datasets/private/form-segmenation/converted/instances_default.json
```


Splitting the dataset
=====================
We split the dataset into training and test sets using the following command:

```shell
python ./cocosplit.py  ~/datasets/private/form-segmenation/converted/instances_default.json ~/datasets/private/form-segmenation/converted/instances_training.json ~/datasets/private/form-segmenation/converted/instances_test.json -s .8
```

Visualizing dataset
===================

```shell
python -m detectron2.data.datasets.coco ~/datasets/private/form-segmenation/converted/instances_training.json ~/datasets/private/form-segmenation/converted/images  dit_dataset 
```

Training the model - for details see README.md
=================================================

Ensure that the `data` folder is a symlink to the `~/datasets/private/form-segmenation/converted/` folder or specific folder for the dataset.

```shell
ln -s ~/datasets/private/form-segmenation/converted data
```


MASKRCNN - BASE MODEL

```shell
python train_net.py --config-file scan_configs/maskrcnn/maskrcnn_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS /mnt/data/marie-ai/model_zoo/unilm/dit/dit_base/dit-base-224-p16-500k-62d53a.pth OUTPUT_DIR ~/tmp/models/dit_scan_detection/tuned-03 SOLVER.IMS_PER_BATCH 1
```


CASCADE

```shell
python train_net.py --config-file scan_configs/cascade/cascade_dit_large.yaml --num-gpus 1 MODEL.WEIGHTS /mnt/data/marie-ai/model_zoo/unilm/dit/dit_base/dit-base-224-p16-500k-62d53a.pth OUTPUT_DIR ~/tmp/models/dit_scan_detection/tuned-04 SOLVER.IMS_PER_BATCH 1
```


```shell
(marie) greg@xpredator:~/dev/unilm$ python ./dit/object_detection/inference.py --image_path ./dit/object_detection/publaynet_example.jpeg --output_file_name output.jpg --config ./dit/object_detection/publaynet_configs/maskrcnn/maskrcnn_dit_base.yaml --opts MODEL.WEIGHTS /home/greg/dev/marieai/marie-ai/model_zoo/unilm/dit/dit-fts/publaynet_dit-b_mrcnn.pth

```

MaskRCNN-
```shell
python inference.py --config-file ./scan_configs/maskrcnn/maskrcnn_dit_large.yaml --image_path ~/tmp/analysis/bad-docs/extracted  --output_path /tmp/dit --opts MODEL.WEIGHTS ~/tmp/models/dit_scan_detection/tuned-02/model_final.pth 
```

CascadeRCNN
```shell
python inference.py --config-file ./scan_configs/cascade/cascade_dit_large.yaml --image_path ~/tmp/analysis/bad-docs/extracted --output_path /tmp/dit --opts MODEL.WEIGHTS ~/tmp/models/dit_scan_detection/tuned-04/model_0001999.pth 
```

## Inference

```shell
python inference.py --config-file ./scan_configs/maskrcnn/maskrcnn_dit_base.yaml --image_path ~/datasets/private/form-segmenation/raw/task_tid-118104-1000/images/segments/tid-118104-1000/PID_790_7528_0_200917639_page_0001.png --output_path output.jpg  --opts MODEL.WEIGHTS ~/tmp/models/dit_scan_detection/tuned-03/model_0001999.pth
```

