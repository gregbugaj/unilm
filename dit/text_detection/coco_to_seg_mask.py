from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


def main():
    coco = COCO(os.path.expanduser('~/datasets/private/resized_scans_dataset/converted/instances_default.json'))
    img_dir = os.path.expanduser('~/datasets/private/resized_scans_dataset/raw/images/')
    converted_img_dir = os.path.expanduser('~/datasets/private/resized_scans_dataset/converted/images/')

    # iterate for each individual annotation and generate masks

    for ann in coco.anns.values():
        img = coco.loadImgs(ann['image_id'])[0]
        im = Image.open(os.path.join(img_dir, img['file_name']))
        im = np.array(im)
        
        if False:
            plt.imshow(im)
            plt.show()
            plt.imshow(coco.annToMask(ann))
            plt.show()

        # save the mask
        mask_dir = '/home/gbugaj/datasets/private/resized_scans_dataset/converted/masks'
        mask = coco.annToMask(ann)

        out_file = os.path.join(mask_dir, img['file_name'])
        base_dir = os.path.dirname(out_file)
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(converted_img_dir, exist_ok=True)

        # Resave the image as a png, this is due to wrong file extension in the original dataset
        im = Image.fromarray(im)
        resave_filename = img['file_name'].split('/')[-1]
        im.save(os.path.join(converted_img_dir, resave_filename))
        
        # convert mask to black and white
        bw_mask = np.zeros_like(mask)
        bw_mask[mask > 0] = 255
        mask = bw_mask

        mask = Image.fromarray(mask).convert('L')
        mask.save(out_file)
        


if __name__ == "__main__":

    main()

# python ./coco_funsd_dit_converter.py --coco_annoations_file ~/datasets/private/resized_scans_dataset/raw/annotations/instances_Train.json --output_file ~/datasets/private/resized_scans_dataset/converted/instances_default.json