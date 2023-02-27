# run inference on a single image using a trained model and save the results to a json file
# Usage: python run_inference.py --image_path /path/to/image.jpg

import argparse
from typing import Iterable, Tuple
import os
import time

import numpy as np
import PIL
import torch
from PIL import Image

from timm.data import create_transform
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data.transforms import str_to_interp_mode
from timm.models import create_model
from timm.models.helpers import load_checkpoint

from torchvision import transforms

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def build_transform():
    args = argparse.Namespace()
    args.input_size = 224
    args.crop_pct = None

    resize_im = args.input_size > 32
    mean = IMAGENET_INCEPTION_MEAN 
    std = IMAGENET_INCEPTION_STD

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=str_to_interp_mode("bicubic")),
            # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)



def preprocess(images: np.ndarray, transform)-> Iterable[torch.Tensor]:
    """Preprocess an image for classification."""
    # image = torch.from_numpy(image).permute(2, 0, 1).float()
    # image = image.unsqueeze(0)
    return [transform(image) for image in images]


def load_model(model_checkpoint_path:str, num_classes:int, device:str):
    model = create_model('deit_base_patch16_224', pretrained=False, num_classes=num_classes, distilled=False)
    load_checkpoint(model, model_checkpoint_path, use_ema=False, strict=False)
    
    model.to(device)
    model.eval()

    return model

def __scale_height(img, target_size, method=Image.Resampling.LANCZOS):
    ow, oh = img.size
    scale = oh / target_size
    w = ow / scale
    h = target_size  # int(max(oh / scale, crop_size))
    resized = img.resize((int(w), int(h)), method)
    return resized

def inference(model, images:list, device:str)-> Tuple[np.ndarray, np.ndarray]:
    """Inference on a batch of images."""
    
    # compute output
    with torch.cuda.amp.autocast(enabled=True):    
        with torch.no_grad():
            # expected input shape: (batch_size, channels, height, width) aka (N, C, H, W)
            images = np.stack(images)
            images = torch.from_numpy(images).to(device, non_blocking=True)
            output = model(images) 
            batch_size = output.shape[0]
            
            # compute softmax and get top k class predictions
            output = torch.nn.functional.softmax(output, dim=1)                   
            topk=(5,)
            maxk = min(max(topk), output.size()[1])
            _, pred = output.topk(maxk, 1, True, True)

            pred = pred.cpu().numpy()
            output = output.cpu().numpy()

            # for each batch, get the output corresponding to the prediction class
            def to_prob(batch_output, batch_pred):
                return [batch_output[batch_pred[i]] for i in range(len(batch_pred))]

            probablities = np.array([to_prob(output[i], pred[i]) for i in range(batch_size)])

            return pred, probablities

def batchify(iterable:Iterable, batch_size:int=1):
    """ batchify is a function that takes in an iterable and returns a generator that yields batches of size batch_size"""
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx:min(ndx + batch_size, l)]


def list_files(image_path:str):
    """ list_files is a function that takes in a path and returns a list of all the files in the path and its subdirectories"""

    # # check if path is a image file
    # def is_image_file(filename)->bool:
    #     return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif", ".tiff"])

    # load images
    image_paths = []
    if os.path.isfile(image_path):
        image_paths.append(image_path)
    elif os.path.isdir(image_path):
        for root, dirs, files in os.walk(image_path):
            for file in files:
                if is_image_file(file):
                    image_paths.append(os.path.join(root, file))
    else:
        raise ValueError("image_path should be a file or directory")

    return image_paths

# get class names from a  file

def get_classes(class_names_path:str)->list[str]:   
    with open(class_names_path, "r") as f:
        classes = f.read().splitlines()
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return classes, class_to_idx

def main(args):
    # load model
    class_names, _ = get_classes(args.class_names_path)
    model = load_model(args.model_path, len(class_names), args.device)
    image_paths = list_files(args.image_path)
    transform = build_transform()
    batches = batchify(image_paths, args.batch_size)


    print(f"Running inference on {len(image_paths)} images")
    
    for batched_paths in batches:
        images = [__scale_height(Image.open(image_path).convert("RGB"), 1000) for image_path in batched_paths]
        images = preprocess(images, transform) 

        start_time = time.time()
        predictions, probablities = inference(model, images, args.device) 
        end_time = time.time()

        print(f"Inference time taken: {end_time - start_time:.4f} seconds")
        
        for image_path, batch_predictions, batch_probablities in zip(batched_paths, predictions, probablities):
            print(f"Predictions for {image_path}:")
            # show the top N predictions
            for pred_class, pred_probability in zip(batch_predictions, batch_probablities):
                print(f"\t>> {class_names[pred_class]}: {pred_class} {pred_probability:.4f}")

            # pred_class = batch_predictions[0]
            # pred_probability = batch_probablities[0]
            # print(f" >> {class_names[pred_class]}: {pred_class} {pred_probability:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--class_names_path", type=str, default="./config/labels.txt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    
    main(args) 