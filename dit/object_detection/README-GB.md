## Training

```
python train_net.py --config-file publaynet_configs/maskrcnn/maskrcnn_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS /home/greg/dev/marieai/marie-ai/model_zoo/unilm/dit/dit-fts/publaynet_dit-b_mrcnn.pth OUTPUT_DIR /home/greg/tmp/models/dit_table_detection/tuned-01 SOLVER.IMS_PER_BATCH 1

```

## Inferencing

```bash
(marie) greg@xpredator:~/dev/unilm$ python ./dit/object_detection/inference.py --image_path ./dit/object_detection/publaynet_example.jpeg --output_file_name output.jpg --config ./dit/object_detection/publaynet_configs/maskrcnn/maskrcnn_dit_base.yaml --opts MODEL.WEIGHTS /home/greg/dev/marieai/marie-ai/model_zoo/unilm/dit/dit-fts/publaynet_dit-b_mrcnn.pth

```

 "categories": [{"supercategory": "", "id": 1, "name": "text"}, {"supercategory": "", "id": 2, "name": "title"}, {"supercategory": "", "id": 3, "name": "list"}, {"supercategory": "", "id": 4, "name": "table"}, {"supercategory": "", "id": 5, "name": "figure"}]}
