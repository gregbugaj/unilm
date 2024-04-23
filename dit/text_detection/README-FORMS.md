
# Convert the dataset

```bash
 python ./coco_funsd_eob_converter.py --coco_annoations_file ~/datasets/private/eob-extract/project_eob-extraction-2023_11_13_17_54_08-coco/annotations/instances_default.json --output_file ~/datasets/private/eob-extract/converted/converted.json


  python ./coco_funsd_dit_converter.py --coco_annoations_file ~/datasets/private/form-segmenation/tid-118104-1000/task/annotations/instances_default.json --output_file ~/datasets/private/form-segmenation/tid-118104-1000/converted/converted.json
```


Splitting the dataset
=====================
We split the dataset into training and test sets using the following command:

```bash
python ./cocosplit.py  ~/datasets/private/resized_scans_dataset/converted/converted.json ~/datasets/private/resized_scans_dataset/converted/instances_training.json ~/datasets/private/resized_scans_dataset/converted/instances_test.json -s .8


python ./cocosplit.py  ~/datasets/private/form-segmenation/tid-118104-1000/converted/converted.json  ~/datasets/private/form-segmenation/tid-118104-1000/converted/instances_training.json ~/datasets/private/form-segmenation/tid-118104-1000/converted/instances_test.json -s .8
```

Training the model
For details see README.md
=======================

ensure that the `data` folder is a symlink to the `~/datasets/private/resized_scans_dataset/converted/` folder or specific folder for the dataset.

```bash
ln -s ~/datasets/private/resized_scans_dataset/converted/ data

ln -s ~/datasets/private/form-segmenation/tid-118104-1000/converted data
```

```
python train_net.py --config-file configs/mask_rcnn_dit_base.yaml --num-gpus 1 --resume MODEL.WEIGHTS ~/models/unilm/dit/dit-base-224-p16-500k-62d53a.pth OUTPUT_DIR ~/resized_scans_detection/tuned-01 SOLVER.IMS_PER_BATCH 1
```


Visualizing dataset

```bash
 python -m detectron2.data.datasets.coco /home/greg/datasets/private/eob-extract/converted/instances_training.json /home/greg/datasets/private/eob-extract/converted/imgs  dit_dataset 
``````


Inference
=========

```
 python ./inference.py --config-file configs/mask_rcnn_dit_base.yaml  --image_path /home/gbugaj/tmp/analysis/scan-of-a-scan  --output_path /tmp/dit --opts  MODEL.WEIGHTS /home/gbugaj/resized_scans_detection/tuned-01/model_0004999.pth
```



# Slicking up the dataset

```bash 
python ./slice_sahi.py
```

Split the sliced dataset and remove empty annotation as this will cause errors in training.


```bash
python ./coco_funsd_dit_converter.py  --coco_annoations_file ~/datasets/private/eob-extract/sliced/sliced_coco.json_coco.json  --output_file ~/datasets/private/eob-extract/sliced/sliced_coco-clean.json
```


```bash
python ./cocosplit.py  ~/datasets/private/eob-extract/sliced/sliced_coco-clean.json  ~/datasets/private/eob-extract/sliced/instances_training.json ~/datasets/private/eob-extract/sliced/instances_test.json -s .8
```
