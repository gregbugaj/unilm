Merge the annotation files from the different datasets.
=======================================================

We merge the annotations from the different datasets into a single one. 

```bash
python ./merge_dir.py --src_dir ~/datasets/funds_dit/raw --output_file ~/datasets/funds_dit/converted/merged.json
```

Converting the dataset.
=======================

The dataset is converted to the DIT format using the following command:

```bash
python ./coco_funsd_dit_converter.py --coco_annoations_file ~/datasets/funds_dit/converted/merged.json --output_file ~/datasets/funds_dit/converted/instances_default.json
```


Splitting the dataset
=====================
We split the dataset into training and test sets using the following command:

```bash
python ./cocosplit.py  ~/datasets/funds_dit/converted/instances_default.json ~/datasets/funds_dit/converted/instances_training.json ~/datasets/funds_dit/converted/instances_test.json -s .8
```



Training the model
For details see README.md
=======================

```bash
 python train_net.py --config-file configs/mask_rcnn_dit_large.yaml --num-gpus 1 --resume MODEL.WEIGHTS /home/greg/tmp/models/dit_text_detection/tuned-2500-LARGE-v2/model_final.pth  OUTPUT_DIR /home/greg/tmp/models/dit_text_detection/tuned-2500-LARGE-v3 SOLVER.IMS_PER_BATCH 1
```
