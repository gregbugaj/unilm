
## Training

```shell
python train_net.py --config-file publaynet_configs/maskrcnn/maskrcnn_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS /home/greg/dev/marieai/marie-ai/model_zoo/unilm/dit/dit-fts/publaynet_dit-b_mrcnn.pth OUTPUT_DIR /home/greg/tmp/models/dit_table_detection/tuned-01 SOLVER.IMS_PER_BATCH 1
```

Train and evaluate on the same dataset
```shell
python train_net.py --resume --eval-only --config-file scan_configs/maskrcnn/maskrcnn_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS /mnt/data/marie-ai/model_zoo/unilm/dit/dit_base/dit-base-224-p16-500k-62d53a.pth OUTPUT_DIR ~/tmp/models/dit_scan_detection/tuned-01 SOLVER.IMS_PER_BATCH 1 
```


## Inference

```shell
python ./dit/object_detection/inference.py --image_path ./dit/object_detection/publaynet_example.jpeg --output_file_name output.jpg --config ./dit/object_detection/publaynet_configs/maskrcnn/maskrcnn_dit_base.yaml --opts MODEL.WEIGHT /home/greg/dev/marieai/marie-ai/model_zoo/unilm/dit/dit-fts/publaynet_dit-b_mrcnn.pth
```

```shell
python inference.py --config-file ./scan_configs/maskrcnn/maskrcnn_dit_base.yaml --image_path ~/datasets/private/scan-of-scan/converted/images/resize_scans/train/189423389_1.png --output_path output.jpg  --opts MODEL.WEIGHTS ~/tmp/models/dit_scan_detection/tuned-01/model_0001999.pth
```

/home/gbugaj/datasets/private/scan-of-scan/TID-118104
## Data preparation


Ensure that the dataset has been updated ot contain the `segmentation` field in the annotations. This is required for the conversion to COCO format.


```shell
python coco_funsd_dit_converter.py --coco_annoations_file ~/datasets/private/eob-registration/task_resize-train-001-2024_01_02_16_50_08-coco/annotations/instances_default.json --output_file ~/datasets/private/eob-registration/converted/converted-scanofscan.json


python coco_funsd_dit_converter.py --coco_annoations_file ~/datasets/private/eob-registration/task_template_master-train-001-2023_12_29_21_14_41-coco/annotations/instances_default.json --output_file ~/datasets/private/eob-registration/converted/converted-templatemaster.json
```

443
226

Depending on the dataset you might have to adjust the `category_id` in the `coco_funsd_dit_converter.py` script. This is the category id that will be used in the COCO annotations. 


# merge datasets

```shell
python ./coco_dataset_merger.py ~/datasets/private/eob-registration/converted/converted-scanofscan.json ~/datasets/private/eob-registration/converted/converted-templatemaster.json   ~/datasets/private/eob-registration/converted/merged.json

```
    
Visualizing dataset

```bash
python -m detectron2.data.datasets.coco ~/datasets/private/eob-registration/converted/merged.json ~/datasets/private/eob-registration/converted/imgs  dit_dataset 
``````
    

Splitting the dataset
=====================
We split the dataset into training and test sets using the following command:

```bash
python ./cocosplit.py  ~/datasets/private/eob-registration/converted/merged.json ~/datasets/private/eob-registration/converted/instances_training.json ~/datasets/private/eob-registration/converted/instances_test.json -s .75
```



## Training

MASKRCNN

```shell
python train_net.py --config-file scan_configs/maskrcnn/maskrcnn_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS /mnt/data/marie-ai/model_zoo/unilm/dit/dit_base/dit-base-224-p16-500k-62d53a.pth OUTPUT_DIR ~/tmp/models/dit_scan_detection/tuned-02 SOLVER.IMS_PER_BATCH 1
```

CASCADE

```shell
python train_net.py --config-file scan_configs/cascade/cascade_dit_large.yaml --num-gpus 1 MODEL.WEIGHTS /mnt/data/marie-ai/model_zoo/unilm/dit/dit_base/dit-base-224-p16-500k-62d53a.pth OUTPUT_DIR ~/tmp/models/dit_scan_detection/tuned-04 SOLVER.IMS_PER_BATCH 1
```

```shell

## Inferencing

```shell
(marie) greg@xpredator:~/dev/unilm$ python ./dit/object_detection/inference.py --image_path ./dit/object_detection/publaynet_example.jpeg --output_file_name output.jpg --config ./dit/object_detection/publaynet_configs/maskrcnn/maskrcnn_dit_base.yaml --opts MODEL.WEIGHTS /home/greg/dev/marieai/marie-ai/model_zoo/unilm/dit/dit-fts/publaynet_dit-b_mrcnn.pth

```

MaskRCNN
```shell
python inference.py --config-file ./scan_configs/maskrcnn/maskrcnn_dit_large.yaml --image_path ~/tmp/analysis/bad-docs/extracted  --output_path /tmp/dit --opts MODEL.WEIGHTS ~/tmp/models/dit_scan_detection/tuned-02/model_final.pth 
```

CascadeRCNN
```shell
python inference.py --config-file ./scan_configs/cascade/cascade_dit_large.yaml --image_path ~/tmp/analysis/bad-docs/extracted --output_path /tmp/dit --opts MODEL.WEIGHTS ~/tmp/models/dit_scan_detection/tuned-04/model_0001999.pth 
```


BLOWS UP MEMORY
192875933 