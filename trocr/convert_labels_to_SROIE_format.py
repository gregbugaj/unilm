import glob
import os
from data import SROIETask2
from tqdm import tqdm
import shutil
import zipfile
from PIL import Image
import random
from sklearn.model_selection import train_test_split


def get_full_image_bounding_box(width: int, height: int):
    x1, y1 = 0, 0  # Top-left corner
    x2, y2 = width, 0  # Top-right corner
    x3, y3 = width, height  # Bottom-right corner
    x4, y4 = 0, height  # Bottom-left corner

    return x1, y1, x2, y2, x3, y3, x4, y4
 
   
def process_dir(slabels_file, src_dir: str, output_dir:str) -> None:

    # read the labels file
    with open(labels_file, 'r') as f:
        labels = f.readlines()
    
    for line in labels:
        try:
            line = line.strip().split(',')
            if len(line) != 3:
                continue
            
            img_file = line[0]
            text = line[1]
            enabled = line[2]            
            if enabled != '1':
                continue

            text = text.upper().strip()
            print(enabled, img_file, text)

            src_image = os.path.join(src_dir, img_file)
            img = Image.open(src_image)
            save_dir = output_dir

            img.save(os.path.join(save_dir, img_file.replace('.jpg', '.jpg')), quality=100)

            width, height = img.size
            label_file = img_file.replace('.jpg', '.txt')

            with open(os.path.join(save_dir, label_file), 'w') as f:
                x1, y1, x2, y2, x3, y3, x4, y4 = get_full_image_bounding_box(width, height)
                f.write(f"{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4},{text}\n")

        except Exception as e:
            print(e)

    return 

    print("Converting images in directory: ", src_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of all image directories
    img_dirs = glob.glob(os.path.join(src_dir, "**"))


    # Split the image directories into training and testing sets
    train_dirs, test_dirs = train_test_split(img_dirs, test_size=0.2, random_state=42)

    for idx, img_dir in enumerate(img_dirs):
        try:
            print("Processing: ", img_dir)
            files = os.listdir(img_dir)
            img_files = [f for f in files if f.endswith('.png')]
            label_file = [f for f in files if f.endswith('.txt')][0]

            print("Label file: ", label_file)
            print("Image files: ", img_files)

            with open(os.path.join(img_dir, label_file), 'r') as f:
                labels = f.readlines()

            text = labels[0].strip()

            for img_file in img_files:
                try:
                    src_image = os.path.join(img_dir, img_file)
                    img = Image.open(src_image)
                    save_dir = output_dir

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

    labels_file= "/home/greg/datasets/SROIE_OCR/ICR_DATA/labels.csv"
    src_dir = '/home/greg/datasets/SROIE_OCR/ICR_DATA/set_1'
    output_dir = '/home/greg/datasets/SROIE_OCR/ICR_DATA/converted'

    
    process_dir(labels_file, src_dir, output_dir)
    