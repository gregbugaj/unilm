
Requires following :

```
pip install deepspeed==0.4.0
pip install timm>=0.6.12

```

# TEXT-DETECTION Training

## Prepare dataset
Additional information about the dataset can be found [README.md](./README.md) file in the root directory of the project.
From withing the project directory run `dit/text_detection` create a symbolid link to the `data` directory (`data -> /home/gbugaj/dataset/funsd_dit`)   


```shell
ln -s ~/datasets/funsd_dit ./data
```


### Prepare your own dataset 

Using MARIE-AI automation CVAT tool, annotate your dataset and export it to COCO format. 
Then run the following command to prepare the dataset for training. 

Expected format of the dataset directory is as follows:

```
./data
├── annotations
│   ├── test
│   └── training
├── dataset
│   ├── testing_data
│   │   └── images
│   └── training_data
├── imgs
├── instances_test.json
└── instances_training.json
```

* `imgs` directory contains images with annotations drawn on them.
* `annotations` directory contains individual COCO annotations in json format.
* `dataset` directory contains images with annotations in COCO format.
* `instances_test.json` and `instances_training.json` are COCO annotations in json format.

Run the following command to prepare the dataset for training:
From `marei-ai/tools` directory run the following command:

```shell
python coco_funds_converter.py
```    


## Training

```shell 
python train_net.py --config-file configs/mask_rcnn_dit_base.yaml --num-gpus 1 --resume MODEL.WEIGHTS /mnt/data/marie-ai/model_zoo/unilm/dit/text_detection/funsd_dit-b_mrcnn.pth  OUTPUT_DIR /tmp/dit/tuned.pth SOLVER.IMS_PER_BATCH 4
```


