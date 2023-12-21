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
python inference.py --config-file ./scan_configs/maskrcnn/maskrcnn_dit_base.yaml --image_path ~/datasets/private/scan-of-scan/converted/images/resize_scans/train/189423389_1.png --output_file_name output.jpg  --opts MODEL.WEIGHTS ~/tmp/models/dit_scan_detection/tuned-01/model_0001999.pth
```


## Data preparation


Ensure that the dataset has been updated ot contain the `segmentation` field in the annotations. This is required for the conversion to COCO format.


```shell
python coco_funsd_dit_converter.py --coco_annoations_file ~/datasets/private/scan-of-scan/12-18-2023/annotations/instances_default.json --output_file ~/datasets/private/scan-of-scan/converted/converted.json

```

Splitting the dataset
=====================
We split the dataset into training and test sets using the following command:

```bash
python ./cocosplit.py  ~/datasets/private/scan-of-scan/converted/converted.json ~/datasets/private/scan-of-scan/converted/instances_training.json ~/datasets/private/scan-of-scan/converted/instances_test.json -s .8
```



## Training

```shell
python train_net.py --config-file scan_configs/maskrcnn/maskrcnn_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS /mnt/data/marie-ai/model_zoo/unilm/dit/dit_base/dit-base-224-p16-500k-62d53a.pth OUTPUT_DIR ~/tmp/models/dit_scan_detection/tuned-01 SOLVER.IMS_PER_BATCH 1

```

## Inferencing

```shell
(marie) greg@xpredator:~/dev/unilm$ python ./dit/object_detection/inference.py --image_path ./dit/object_detection/publaynet_example.jpeg --output_file_name output.jpg --config ./dit/object_detection/publaynet_configs/maskrcnn/maskrcnn_dit_base.yaml --opts MODEL.WEIGHTS /home/greg/dev/marieai/marie-ai/model_zoo/unilm/dit/dit-fts/publaynet_dit-b_mrcnn.pth

```

MaskRCNN
python inference.py --config-file ./scan_configs/maskrcnn/maskrcnn_dit_large.yaml --image_path ~/tmp/analysis/bad-docs/extracted/180857073-0003.tif  --output_file_name output.jpg --opts MODEL.WEIGHTS ~/tmp/models/dit_scan_detection/tuned-02/model_final.pth 


CascadeRCNN
python inference.py --config-file ./scan_configs/cascade/cascade_dit_large.yaml --image_path ~/tmp/analysis/bad-docs/extracted/180857073-0003.tif  --output_file_name output-cascade.jpg --opts MODEL.WEIGHTS ~/tmp/models/dit_scan_detection/tuned-03/model_0021999.pth 



BLOWS UP MEMORY
192875933 