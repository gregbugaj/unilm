import argparse
import os
import math
import random
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from typing import List, Tuple, Dict, Any, Optional, Union



def __scale_height(img, target_size, method=Image.Resampling.LANCZOS):
    ow, oh = img.size
    scale = oh / target_size
    w = ow / scale
    h = target_size  # int(max(oh / scale, crop_size))
    resized = img.resize((int(w), int(h)), method)
    return resized

def convert_folder_to_rvlcdip(source_dir, output_dir, train_size:float, validation_size:float, test_size:float):
    """
    Convert a folder of images and subdirectories to a single directory of images with a single csv file containing the image path and the label
    """
    print("Converting {} to {}".format(source_dir, output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # ensure that train + validation + test = 1.0
    if train_size + validation_size + test_size != 1.0:
        print("Error: train + validation + test must equal 1.0")

    files_by_class = {}
    all_files = []

    for root, dirs, files in os.walk(source_dir):
        clazz = os.path.basename(root)
        for file in files:
            files_by_class[clazz] = files_by_class.get(clazz, []) + [os.path.join(root, file)]
            all_files.append(os.path.join(root, file))

    class_ids = {clazz: i for i, clazz in enumerate(sorted(files_by_class.keys()))}

    # shuffle files and split into train, validation, and test sets by class 
    def shuffle_and_split(files:list, val_size:float, test_size:float):
        random.shuffle(files)
        size = len(files)
        validation_size = int(math.ceil(size * val_size))
        test_size = int(math.ceil(size * test_size))
        training_size = int(size - val_size - test_size)

        print("Class {} has {} files, training = {} validation = {} test = {} ".format(clazz, size, training_size, validation_size, test_size))
        return files[:training_size], files[training_size:training_size+validation_size], files[training_size+validation_size:]

    training_files = []
    validation_files = []
    testing_files = []

    for clazz, files in files_by_class.items():
        print("Class {} has {} files".format(clazz, len(files)))
        training_files_by_class, validation_files_by_class, test_files_by_class = shuffle_and_split(files, validation_size, test_size)

        training_files += training_files_by_class
        validation_files += validation_files_by_class
        testing_files += test_files_by_class


    random.shuffle(training_files)
    random.shuffle(validation_files)
    random.shuffle(testing_files)

    if False:
        all_files = random.sample(all_files, len(all_files))
        size = len(all_files)

        validation_size = math.ceil(size * validation_size)  # 5 percent validation size
        test_size = math.ceil(size * test_size)  # 25 percent testing size
        training_size = size - validation_size - test_size  # 70 percent training


        print("Class >>  size = {} training = {} validation = {} test = {} ".format(size, training_size, validation_size, test_size))
            
        validation_files = all_files[:validation_size]
        testing_files = all_files[validation_size : validation_size+test_size]
        training_files = all_files[validation_size+test_size:]

    print("Number of training images   : {}".format( len(training_files)))
    print("Number of validation images : {}".format(len(validation_files)))
    print("Number of testing images    : {}".format(len(testing_files)))
    
    # os.exit()
    # print class distribution
    print("Class distribution:")
    
    for i, clazz in enumerate(sorted(files_by_class.keys())):
        print(f"{clazz} [{class_ids[clazz]}] = {len(files_by_class[clazz])}")

    print("Class ids: {}".format(class_ids))

    # prepare output directories
    image_dir_out = os.path.join(output_dir, 'images')
    labels_dir_out = os.path.join(output_dir, 'labels')

    Path(image_dir_out).mkdir(parents=True, exist_ok=True)
    Path(labels_dir_out).mkdir(parents=True, exist_ok=True)

    # write labels file
    with open(os.path.join(labels_dir_out, "labels.txt"), 'w') as f:
        for clazz in sorted(files_by_class.keys()):
            f.write("{} {}\n".format(clazz, class_ids[clazz]))

    # write training files
    def copyfiles(files,  destDir, split_filename:str):
        with open(os.path.join(labels_dir_out, split_filename), 'w') as f:
            for filename in files:
                try:
                    # get parent directory
                    clazz = os.path.dirname(filename).split(os.sep)[-1]
                    dest_filename = os.path.join(destDir, clazz, os.path.basename(filename))
                    rel_dest_filename = os.path.relpath(filename, source_dir)
                    # print("Copying {} to {}".format(filename, dest_filename))

                    Path(os.path.dirname(dest_filename)).mkdir(parents=True, exist_ok=True)
                    shutil.copy(filename, dest_filename)
                    
                    # print("Copying {} to {}".format(filename, dest_filename))
                    img = Image.open(dest_filename)
                    framed = __scale_height(img, 1000, method=Image.LANCZOS)
                    framed.save(dest_filename)
                    f.write("{} {}\n".format(rel_dest_filename, class_ids[clazz]))
                except Exception as e:
                    print("Error copying file {}: {}".format(filename, e))


    copyfiles(testing_files, image_dir_out, "test.txt")
    copyfiles(validation_files, image_dir_out, "val.txt")
    copyfiles(training_files, image_dir_out, "train.txt")


if __name__ == "__main__":  
    parser = argparse.ArgumentParser()  
    parser.add_argument("--source_dir", type=str, required=True)  
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--validation_size", type=float, required=True, default=0.0)
    parser.add_argument("--test_size", type=float, required=True, default=0.2)
    parser.add_argument("--train_size", type=float, required=True, default=0.8)

    args = parser.parse_args()  

    convert_folder_to_rvlcdip(args.source_dir, args.output_dir, args.train_size, args.validation_size, args.test_size)


