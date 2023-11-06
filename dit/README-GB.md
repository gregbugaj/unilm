
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

python train_net.py --config-file configs/mask_rcnn_dit_base.yaml --num-gpus 1 --resume MODEL.WEIGHTS /mnt/data/marie-ai/model_zoo/unilm/dit/text_detection/td-syn_dit-b_mrcnn.pth  OUTPUT_DIR /home/greg/tmp/models/dit_text_detection/tuned.pth SOLVER.IMS_PER_BATCH 2



python train_net.py --config-file configs/mask_rcnn_dit_large.yaml --num-gpus 4 --resume MODEL.WEIGHTS ~/model_zoo/unilm/dit/text_detection/td-syn_dit-l_mrcnn.pth  OUTPUT_DIR ~/models/dit_text_detection/tuned-01 SOLVER.IMS_PER_BATCH 8


TRAINS-002
~/dev/unilm/dit/text_detection$ 
python train_net.py --config-file configs/mask_rcnn_dit_large.yaml --num-gpus 4 --resume MODEL.WEIGHTS ~/model_zoo/unilm/dit/text_detection/td-syn_dit-l_mrcnn.pth  OUTPUT_DIR ~/models/dit_text_detection/tuned-01 SOLVER.IMS_PER_BATCH 8Connection to 172.83.14.99 closed by remote host.



python train_net.py --config-file configs/mask_rcnn_dit_large.yaml --num-gpus 1 --resume MODEL.WEIGHTS ~/model_zoo/unilm/dit/text_detection/td-syn_dit-l_mrcnn.pth  OUTPUT_DIR ~/models/dit_text_detection/tuned-01 SOLVER.IMS_PER_BATCH 3
```

## update the co

python ./coco_funsd_dit_converter.py  --coco_annoations_file ./data/instances_training.json 


# Timings
A100 32GB
[09/23 03:13:19 d2.evaluation.evaluator]: Total inference time: 0:04:02.963073 (8.378037 s / iter per device, on 4 devices)
[09/23 03:13:19 d2.evaluation.evaluator]: Total inference pure compute time: 0:00:18 (0.624468 s / iter per device, on 4 devices)

4090 24GB
[09/22 21:49:22 d2.evaluation.evaluator]: Total inference time: 0:04:25.386144 (2.057257 s / iter per device, on 1 devices)
[09/22 21:49:22 d2.evaluation.evaluator]: Total inference pure compute time: 0:00:32 (0.253688 s / iter per device, on 1 devices)


 rsync -av --progress /home/greg/dev/marieai/marie-ai/model_zoo/unilm/dit/text_detection/td-syn_dit-l_mrcnn.pth  paperspace@172.83.14.99:/home/paperspace/model_zoo/unilm/dit/text_detection


  rsync -av --progress /home/greg/dev/marieai/marie-ai/model_zoo/unilm/dit/text_detection/td-syn_dit-l_mrcnn.pth  paperspace@172.83.14.99:/home/paperspace/model_zoo/unilm/dit/text_detection