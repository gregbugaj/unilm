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
                src_image = os.path.join(img_dir, img_file)
                img = Image.open(src_image)
                save_dir = output_dir

                img.save(os.path.join(save_dir, img_file.replace('.png', '.jpg')), quality=100)

                width, height = img.size
                label_file = img_file.replace('.png', '.txt')

                with open(os.path.join(save_dir, label_file), 'w') as f:
                    x1, y1, x2, y2, x3, y3, x4, y4 = get_full_image_bounding_box(width, height)
                    f.write(f"{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4},{text}\n")

            print(f"Processed: {img_dir}")
        except Exception as e:
            print(e)

def split_dir(src_dir, output_dir):
    
    # Create directories for the training and testing sets
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    img_dirs = glob.glob(os.path.join(src_dir, "*.jpg"))
    random.shuffle(img_dirs)

    train_dirs, test_dirs = train_test_split(img_dirs, test_size=0.2, random_state=42)
    # Copy the images and labels to the training and testing directories
    for img_dir in train_dirs:
        shutil.copy(img_dir, train_dir)
        shutil.copy(img_dir.replace('.jpg', '.txt'), train_dir)

    for img_dir in test_dirs:
        shutil.copy(img_dir, test_dir)
        shutil.copy(img_dir.replace('.jpg', '.txt'), test_dir)
    


if __name__ == "__main__":

    src_dir = '/tmp/boxes/number'
    output_dir = '/home/greg/datasets/SROIE_OCR/converted'
    output_train_test_dir = '/home/greg/datasets/SROIE_OCR/ready'

    # process_dir(src_dir, output_dir)

    split_dir(output_dir, output_train_test_dir)
    

