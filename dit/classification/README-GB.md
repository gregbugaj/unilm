# Folder to RVLCDIP Dataset conversion

## convert the RVLCDIP Dataset to the format rquired format


```bash
python ./converter.py --source_dir /home/greg/datasets/private/data-hipa/medical_page_classification/raw --output_dir /home/greg/datasets/private/data-hipa/medical_page_classification/output/ --validation_size 0  --test_size .2 --train_size .8
```


# fine tune the model on the converted dataset

```bash
python ./run_class_finetuning.py
```


# run inference on the converted dataset

```bash
python ./run_inference.py --image_path /home/greg/datasets/dataset/data-hipa/medical_page_classification/output/images/ACH-CHECK/169689764_16.tiff --model_path output_dir_pages/checkpoint-best.pth

```    

# Notes

```
(marie) greg@xpredator:~/dev/unilm/dit/classification$ python ./run_inference.py --image_path /home/greg/datasets/dataset/data-hipa/medical_page_classification/output/images/PATPAY/159559786_2.png  --model_path output_dir_pages/checkpoint-best.pth
Running inference on 1 images
Inference time taken: 1.2363 seconds
Predictions for /home/greg/datasets/dataset/data-hipa/medical_page_classification/output/images/PATPAY/159559786_2.png:
	>> PATPAY: 9 0.7314
	>> CHECK: 2 0.0998
	>> MONEY-ORDER: 7 0.0333
	>> CORRESPONDENCE: 3 0.0274
	>> EOB: 6 0.0227
(marie) greg@xpredator:~/dev/unilm/dit/classification$ python ./run_inference.py --image_path /home/greg/datasets/dataset/data-hipa/medical_page_classification/output/images/MONEY-ORDER/159743451_0.png  --model_path output_dir_pages/checkpoint-best.pth
Running inference on 1 images
Inference time taken: 1.1750 seconds
Predictions for /home/greg/datasets/dataset/data-hipa/medical_page_classification/output/images/MONEY-ORDER/159743451_0.png:
	>> CHECK: 2 0.7992
	>> MONEY-ORDER: 7 0.0907
	>> PATPAY: 9 0.0491
	>> CORRESPONDENCE: 3 0.0102
	>> SUMMARY: 11 0.0089
(marie) greg@xpredator:~/dev/unilm/dit/classification$ python ./run_inference.py --image_path /home/greg/datasets/dataset/data-hipa/medical_page_classification/output/images/ACH-CHECK/169327458_0.tiff --model_path output_dir_pages/checkpoint-best.pth
Running inference on 1 images
Inference time taken: 1.1952 seconds
Predictions for /home/greg/datasets/dataset/data-hipa/medical_page_classification/output/images/ACH-CHECK/169327458_0.tiff:
	>> EOB: 6 0.7625
	>> SUMMARY: 11 0.0795
	>> CORRESPONDENCE: 3 0.0599
	>> PATPAY: 9 0.0317
	>> SUBSTITUTION-DOC: 10 0.0104

```