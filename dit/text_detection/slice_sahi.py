import os
os.getcwd()


from sahi.slicing import slice_coco
from sahi.utils.file import load_json

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# https://github.com/obss/sahi/blob/main/demo/slicing.ipynb

base_dir = os.path.expanduser("~/datasets/private/eob-extract/converted")
base_img_dir = os.path.join(base_dir, "imgs")
slice_dir = os.path.join(base_img_dir, "sliced")
coco_dict = load_json(os.path.join(base_dir, "converted.json"))

f, axarr = plt.subplots(1, 1, figsize=(12, 12))
# read image
img_ind = 0
img = Image.open(os.path.join(base_img_dir, coco_dict["images"][img_ind]["file_name"])).convert('RGBA')
# iterate over all annotations
for ann_ind in range(len(coco_dict["annotations"])):
    xywh = coco_dict["annotations"][ann_ind]["bbox"]
    xyxy = [xywh[0], xywh[1], xywh[0]+xywh[2], xywh[1]+xywh[3]]
    # visualize bbox over image
    ImageDraw.Draw(img, 'RGBA').rectangle(xyxy, width=5)

    break
axarr.imshow(img)
# display axarr in the terminal
# plt.show()
plt.savefig('grid.png')

# qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
# python -m pip install PyQt5
# export QT_QPA_PLATFORM=offscreen


coco_dict, coco_path = slice_coco(
    coco_annotation_file_path= os.path.join(base_dir, "converted.json"),
    image_dir=base_img_dir,
    output_coco_annotation_file_name="sliced_coco.json",
    ignore_negative_samples=False,
    output_dir=slice_dir,
    slice_height=768,
    slice_width=768,
    overlap_height_ratio=0.25,
    overlap_width_ratio=0.25,
    min_area_ratio=0.1,
    verbose=True
)


f, axarr = plt.subplots(4, 5, figsize=(13,13))
img_ind = 0
for row_ind in range(4):
    for column_ind in range(5):
        # read image
        img = Image.open( os.path.join(slice_dir, coco_dict["images"][img_ind]["file_name"])).convert('RGBA')
        # iterate over all annotations
        for ann_ind in range(len(coco_dict["annotations"])):
            # find annotations that belong the selected image
            if coco_dict["annotations"][ann_ind]["image_id"] == coco_dict["images"][img_ind]["id"]:
                # convert coco bbox to pil bbox
                xywh = coco_dict["annotations"][ann_ind]["bbox"]
                xyxy = [xywh[0], xywh[1], xywh[0]+xywh[2], xywh[1]+xywh[3]]
                # visualize bbox over image
                ImageDraw.Draw(img, 'RGBA').rectangle(xyxy, width=5)
            axarr[row_ind, column_ind].imshow(img)
        img_ind += 1

    break

# display axarr in the terminal
# plt.show()
plt.savefig('annoation.png')