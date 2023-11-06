import argparse
import io
import json
import multiprocessing as mp
import os
import numpy as np
import copy

from concurrent.futures.thread import ThreadPoolExecutor
from functools import lru_cache
from multiprocessing import Pool

# https://github.com/facebookresearch/detectron2/issues/485
def process_split(coco_annoations_file:str):

    with io.open(coco_annoations_file, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    # loop over the annotations and extract the bounding boxes
    for annotation in data["annotations"]:
        x,y,w,h = annotation['bbox']
        annotation['segmentation'] = [[x,y, x+w,y, x,y+h, x+w,y+h]]
        annotation['area'] = w*h

    # split data inot training and testing sets
    split_percentage = 0.8

    images = data["images"]

    total_count = len(images)
    sample_count = int(total_count * split_percentage)
    print(f"split_percentage = {split_percentage}")
    print(f"total_count      = {total_count}")
    print(f"sample_count     = {sample_count}")

    # np.random.shuffle(images)

    train_set = images[0:sample_count]
    test_set = images[sample_count:]

    print(f"Train size : {len(train_set)}")
    print(f"Test size : {len(test_set)}")

    # deep copy the data
    # clone the data
    
    train_data = copy.deepcopy(data) 
    test_data = copy.deepcopy(data) 

    train_data['images'] = train_set
    test_data['images']  = test_set
    
    # save the annotations to a new file
    with open(f"/tmp/instances_training.json", 'w') as outfile:
        json.dump(train_data, outfile)

    with open(f"/tmp/instances_test.json", 'w') as outfile:
        json.dump(test_data, outfile)


# https://github.com/facebookresearch/detectron2/issues/485
def process(coco_annoations_file:str, output_file:str):

    with io.open(coco_annoations_file, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    # loop over the annotations and ensure that the segmentation node is present and the area is set
    for i in range(len(data['annotations'])):   
        annotation = data['annotations'][i]
        img_id = data['annotations'][i]['image_id']

        x0, y0, w, h = annotation['bbox']
        x1, y1 = x0 + w, y0 + h
        polygon = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        annotation['segmentation'] = polygon
        # annotation['segmentation'] = [[x,y, x+w,y, x,y+h, x+w,y+h]]


        annotation['area'] = w*h
    
    # need to loop over images and check if the image has a corresponding annotation entry if not remove it or it will cause an error in the training
    img2id, id2bbox = {}, {}
    updated_images = []

    for i in range(len(data['images'])):
        key = os.path.basename(data['images'][i]['file_name'][:-len('.png')])
        assert key not in img2id.keys()
        img2id[key] = data['images'][i]['id']
        has_annotation = False

        for j in range(len(data['annotations'])):
            if data['annotations'][j]['image_id'] == img2id[key]:
                has_annotation = True
                break
        if has_annotation:
            updated_images.append(data['images'][i])
        else:
            print(f"Annotation not found for image {key}")


    data['images'] = updated_images

    # validate the annotations data structure
    # This is base don the funds_evaluation.py script
    id2img = {}
    gt = {}
    for img in data['images']:
        id = img['id']
        name = os.path.basename(img['file_name'])[:-len('.jpg')]
        assert id not in id2img.keys()
        id2img[id] = name
    assert len(id2img) == len(data['images'])    

    img2id, id2bbox = {}, {}
    for i in range(len(data['images'])):
        key = os.path.basename(data['images'][i]['file_name'][:-len('.png')])
        assert key not in img2id.keys()
        img2id[key] = data['images'][i]['id']

    for i in range(len(data['annotations'])):
        img_id = data['annotations'][i]['image_id']
        if img_id not in id2bbox.keys():
            id2bbox[img_id] = []
        x0, y0, w, h = data['annotations'][i]['bbox']
        x1, y1 = x0 + w, y0 + h
        line = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        # print(f"img_id = {img_id} line = {line}")
        id2bbox[img_id].append(
            {
                'points': line,
                'text': 1234,
                'ignore': False,
            }
        )

    print(f"len(id2img) = {len(id2img)}")
    print(f"len(img2id) = {len(img2id)}")

    for key, val in img2id.items():
        assert key not in gt.keys()
        gt[key] = id2bbox[val]

    with open(output_file, 'w', encoding='utf-8') as f:            
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts COCO annotations to DIT format")
    parser.add_argument(
        "--coco_annoations_file",
        type=str,
        help="Path to the COCO annotations file",
        required=True,
    )

    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to the output file",
        required=True,
    )
    
    args = parser.parse_args()

    process(args.coco_annoations_file, args.output_file)    

