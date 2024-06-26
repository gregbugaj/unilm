import argparse
import io
import json
import math
import multiprocessing as mp
import os
import numpy as np
import copy

from concurrent.futures.thread import ThreadPoolExecutor
from functools import lru_cache
from multiprocessing import Pool


# https://github.com/facebookresearch/detectron2/issues/485
def process(coco_annoations_file:str, output_file:str):

    with io.open(coco_annoations_file, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    
    print("Total annotations = ", len(data['annotations']))
    id_to_remove = []
    id_to_keep = []
    # loop over categories and remove the ones that are not text
    for i in range(len(data['categories'])):
        name = data['categories'][i]['name']        
        marker = name.split('.')[0]
        # "r", "s", "d"
        # if marker not in ["m"] or name.endswith('_answer') or name != 'm.claim_data':
        if name != 'r.patient_name':
            id_to_remove.append(data['categories'][i]['id'])
            data['categories'][i] = None
            continue
        
        id_to_keep.append(data['categories'][i]['id'])
        print( name )

        # if data['categories'][i]['name'] != 'text':
        #     data['categories'].pop(i)
    print("Total categories = ", len(data['categories']))
    print("Total id_to_remove = ", len(id_to_remove))
    print("Total id_to_keep = ", len(id_to_keep))

    print(id_to_keep)
    #remove the categories that are not text
    data['categories'] = [x for x in data['categories'] if x is not None]
    
    print("Total categories = ", len(data['categories'])) 
    
    # loop over the annotations and ensure that the segmentation node is present and the area is set    
    for i in range(len(data['annotations'])):   
        ann = data['annotations'][i]

        # x1, y1, x2, y2 = ann['bbox']
        # x = max(0, min(math.floor(x1), math.floor(x2)))
        # y = max(0, min(math.floor(y1), math.floor(y2)))
        # w, h = math.ceil(abs(x2 - x1)), math.ceil(abs(y2 - y1))
        x, y, w, h = ann['bbox']
        bbox = [x, y, w, h]
        segmentation = [x, y, x + w, y, x + w, y + h, x, y + h]
        
        ann['bbox'] = bbox
        ann['segmentation'] = [segmentation]
        ann['area'] = w * h

        if ann['category_id'] not in id_to_keep:
            print("Removing annotation with id = ", ann['id'])
            data['annotations'][i] = None

    print("Total annotations = ", len(data['annotations']))
    #remove the annotations that are not text
    data['annotations'] = [x for x in data['annotations'] if x is not None]
    
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

    # save the annotations to a new file
    with open(output_file, 'w', encoding='utf-8') as f:            
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    # ref https://github.com/open-mmlab/mmocr/blob/main/tools/dataset_converters/textdet/funsd_converter.py
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


