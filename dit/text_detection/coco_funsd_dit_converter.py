import argparse
import concurrent.futures
import distutils.util
import glob
import hashlib
import io
import json
import logging
from rich.logging import RichHandler
import multiprocessing as mp
import os
import random
import shutil
import string
import sys
import time
import uuid
import numpy as np

from concurrent.futures.thread import ThreadPoolExecutor
from functools import lru_cache
from multiprocessing import Pool

# https://github.com/facebookresearch/detectron2/issues/485
def process(coco_annoations_file:str):

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
    import copy

    train_data =copy.deepcopy(data) 
    test_data = copy.deepcopy(data) 

    train_data['images'] = train_set
    test_data['images']  = test_set
    
    # print(f"Train size : {len(train_data['images'])}")
    # print(f"Test size : {len(test_data['images'])}")

    # save the annotations to a new file
    with open(f"/tmp/instances_training.json", 'w') as outfile:
        json.dump(train_data, outfile)

    with open(f"/tmp/instances_test.json", 'w') as outfile:
        json.dump(test_data, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts COCO annotations to DIT format")
    parser.add_argument(
        "--coco_annoations_file",
        type=str,
        help="Path to the COCO annotations file",
        required=True,
    )

    args = parser.parse_args()

    process(args.coco_annoations_file)
