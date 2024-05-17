import glob
import os
from data import SROIETask2
from tqdm import tqdm
import shutil
import zipfile
from PIL import Image
import random


def get_full_image_bounding_box(width: int, height: int):
    x1, y1 = 0, 0  # Top-left corner
    x2, y2 = width, 0  # Top-right corner
    x3, y3 = width, height  # Bottom-right corner
    x4, y4 = 0, height  # Bottom-left corner

    return x1, y1, x2, y2, x3, y3, x4, y4
 
   
from sklearn.model_selection import train_test_split

def process_dir(src_dir: str, output_dir:str) -> None:
    print("Converting images in directory: ", src_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of all image directories
    img_dirs = glob.glob(os.path.join(src_dir, "**"))


    for idx, img_dir in enumerate(img_dirs):
        try:
            print("Processing: ", img_dir)
            files = os.listdir(img_dir)
            img_files = [f for f in files if f.endswith('.png')]

            print("Image files: ", img_files)


            for img_file in img_files:
                try:
                    print("img_file: ", img_file)
                    src_image = os.path.join(img_dir, img_file)
                    img = Image.open(src_image)

                    label_file = img_file.replace('.png', '.txt')
                    with open(os.path.join(img_dir, label_file), 'r') as f:
                        labels = f.readlines()

                    text = labels[0].strip()
                    save_dir = output_dir

                    if text == '':
                        raise Exception(f"Empty text found in label file. : {label_file}")
                    
                    # Convert image to RGB if it's RGBA
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')

                    img.save(os.path.join(save_dir, img_file.replace('.png', '.jpg')), quality=100)

                    width, height = img.size
                    label_file = img_file.replace('.png', '.txt')

                    with open(os.path.join(save_dir, label_file), 'w') as f:
                        x1, y1, x2, y2, x3, y3, x4, y4 = get_full_image_bounding_box(width, height)
                        f.write(f"{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4},{text}\n")
                except Exception as e:
                            print(e)

            print(f"Processed: {img_dir}")
        except Exception as e:
            print(e)

 

if __name__ == "__main__":

    src_dir = '/home/greg/datasets/SROIE_OCR/lines/raw'
    # src_dir = '/home/greg/datasets/SROIE_OCR/raw'
    output_dir = '/home/greg/datasets/SROIE_OCR/lines/converted'

    
    process_dir(src_dir, output_dir)
    

