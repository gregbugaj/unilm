Merge the annotation files from the different datasets.
=======================================================

We merge the annotations from the different datasets into a single one. 

```bash
python ./merge_dir.py --src_dir ~/datasets/funsd_dit/raw --output_file ~/datasets/funsd_dit/converted/merged.json
```

Converting the dataset.
=======================

The dataset is converted to the DIT format using the following command:

```bash
python ./coco_funsd_dit_converter.py --coco_annoations_file ~/datasets/funsd_dit/converted/merged.json --output_file ~/datasets/funsd_dit/converted/instances_default.json
```


Splitting the dataset
=====================
We split the dataset into training and test sets using the following command:

```bash
python ./cocosplit.py  ~/datasets/funsd_dit/converted/instances_default.json ~/datasets/funsd_dit/converted/instances_training.json ~/datasets/funsd_dit/converted/instances_test.json -s .8
```



Training the model
For details see README.md
=======================

ensure that the `data` folder is a symlink to the `datasets/funsd_dit/converted` folder or specific folder for the dataset.

```
ln -s ~/datasets/funsd_dit/converted data

data -> /home/greg/datasets/funsd_dit/v4/
```

Local 1 GPU


```bash
 python train_net.py --config-file configs/mask_rcnn_dit_large.yaml --num-gpus 1 --resume MODEL.WEIGHTS /mnt/data/marie-ai/model_zoo/unilm/dit/text_detection/tuned-4000-LARGE/model_final.pth  OUTPUT_DIR  /mnt/data/marie-ai/model_zoo/unilm/dit/text_detection/tuned-4000-20240405-002  SOLVER.IMS_PER_BATCH 1


 python train_net.py --config-file configs/mask_rcnn_dit_large.yaml --num-gpus 1 --resume MODEL.WEIGHTS ~/dev/marieai/marie-ai/model_zoo/unilm/dit/text_detection/td-syn_dit-l_mrcnn.pth  OUTPUT_DIR ~/tmp/models/dit_text_detection/fixed-segmenation SOLVER.IMS_PER_BATCH 1

```

```bash
 python ./inference.py --config-file configs/mask_rcnn_dit_large.yaml  --image_path ~/datasets/private/medical_page_classification/raw/EOB  --output_path /tmp/dit --opts  MODEL.WEIGHTS /mnt/data/marie-ai/model_zoo/unilm/dit/text_detection/tuned-4000-20240405-002/model_0000999.pth
```

A-100 4-GPU

```bash
python train_net.py --config-file configs/mask_rcnn_dit_large.yaml --num-gpus 4 --resume MODEL.WEIGHTS ~/model_zoo/unilm/dit/text_detection/td-syn_dit-l_mrcnn.pth  OUTPUT_DIR ~/models/dit_text_detection/tuned-01 SOLVER.IMS_PER_BATCH 8

python train_net.py --config-file configs/mask_rcnn_dit_large.yaml --num-gpus 4 --resume MODEL.WEIGHTS ~/models/dit_text_detection/tuned-01/model_0022999.pth  OUTPUT_DIR ~/models/dit_text_detection/tuned-02 SOLVER.IMS_PER_BATCH 8
```

A-100 1 GPU

```bash
python train_net.py --config-file configs/mask_rcnn_dit_large.yaml --num-gpus 1 --resume MODEL.WEIGHTS ~/model_zoo/unilm/dit/text_detection/td-syn_dit-l_mrcnn.pth  OUTPUT_DIR ~/models/dit_text_detection/tuned-01 SOLVER.IMS_PER_BATCH 4

python train_net.py --config-file configs/mask_rcnn_dit_large.yaml --num-gpus 1 --resume MODEL.WEIGHTS ~/models/dit_text_detection/tuned-01/model_0048999.pth OUTPUT_DIR ~/models/dit_text_detection/tuned-02 SOLVER.IMS_PER_BATCH 4
```    

------------------------------------------------------------------------------------------------------------------------
# Table Detection

## prepare the dataset


```bash
python ./merge_dir.py --src_dir ~/datasets/private/corr-tables/annotations --output_file ~/datasets/private/corr-tables/converted/merged.json

```

```bash
python ./coco_funsd_dit_converter.py --coco_annoations_file ~/datasets/private/corr-tables/converted/merged.json --output_file ~/datasets/private/corr-tables/converted/instances_default.json
```

```bash
python ./cocosplit.py  ~/datasets/private/corr-tables/converted/instances_default.json ~/datasets/private/corr-tables/converted/instances_training.json ~/datasets/private/corr-tables/converted/instances_test.json -s .8
```


```bash 
python train_net.py --config-file configs/mask_rcnn_dit_large.yaml --num-gpus 1 --resume MODEL.WEIGHTS ~/dev/marieai/marie-ai/model_zoo/unilm/dit/text_detection/td-syn_dit-l_mrcnn.pth  OUTPUT_DIR ~/models/dit_text_detection/tuned-01 SOLVER.IMS_PER_BATCH 1


```


data -> /home/greg/datasets/funsd_dit/v4/


# visualize the dataset

```bash
python -m detectron2.data.datasets.coco ~/datasets/funsd_dit/v6/instances_training.json ~/datasets/funsd_dit/v6/imgs  dit_dataset_v6
```


python ./cocosplit.py  ./data/instances_default.json ./data/instances_training.json ./data/instances_test.json -s .8