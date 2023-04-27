import os
import copy
import io
import json
import logging
import os
import random
import shutil
import uuid
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from typing import List, Tuple, Dict, Any, Optional, Union



def __scale_height(img, target_size, method=Image.LANCZOS)->Tuple[Image, Tuple[int, int]]:
    ow, oh = img.size
    scale = oh / target_size
    w = ow / scale
    h = target_size  # int(max(oh / scale, crop_size))
    resized = img.resize((int(w), int(h)), method)
    return resized, resized.size


def main():
    name = "train"
    root_dir = ""
    
    src_dir = os.path.join(root_dir, f"{name}deck-raw-01")
    dst_path = os.path.join(root_dir, "dataset", f"{name}ing_data")

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for file in os.listdir(src_dir):
        if file.endswith(".jpg"):
            img = Image.open(os.path.join(src_dir, file))
            img, size = __scale_height(img, 1000)
            img.save(os.path.join(dst_path, file))
            
                
if __name__ == "__main__":
    main()


