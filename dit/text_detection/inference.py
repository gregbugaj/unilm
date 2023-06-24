import argparse
import logging
import os
import sys
import torch
import cv2

from ditod import add_vit_config

from detectron2.utils.visualizer import ColorMode, Visualizer

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.engine import DefaultPredictor

from ditod import MyTrainer, add_vit_config

import argparse

def setup_cfg(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # add_coat_config(cfg)
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device

    cfg.freeze()
    # default_setup(cfg, args)
    return cfg


def build_model_from_config(cfg):
    """
    Returns:
        torch.nn.Module:

    It now calls :func:`detectron2.modeling.build_model`.
    Overwrite it if you'd like a different model.
    """
    model = build_model(cfg)
    logger = logging.getLogger(__name__)
    logger.info("Model:\n{}".format(model))
    return model

def main(args):
    print("Inference")
    # Step 1: instantiate config
    cfg = setup_cfg(args)
    print(cfg)
    model = build_model_from_config(cfg)    
    # Step 4: define model
    predictor = DefaultPredictor(cfg)

    # Step 5: run inference
    img = cv2.imread(args.image_path)
    print(img)
    output = predictor(img)["instances"]
    md=None

    print(output)

    v = Visualizer(img[:, :, ::-1],
                    md,
                    scale=1.0,
                    instance_mode=ColorMode.SEGMENTATION)
    result = v.draw_instance_predictions(output.to("cpu"))
    result_image = result.get_image()[:, :, ::-1]

    # # step 6: save
    cv2.imwrite("/tmp/dit/result.png", result_image)
    # cv2.imwrite(args.output_file_name, result_image)


### Sample usage
###  python ./inference.py --config-file configs/mask_rcnn_dit_base.yaml  --image_path /home/gbugaj/tmp/2022-08-09/PID_1549_8460_0_159274481.tif --output_path /tmp/dit --opts  MODEL.WEIGHTS /home/gbugaj/models/unilm/dit/text_detection/td-syn_dit-b_mrcnn.pth

def get_parser():
    parser = argparse.ArgumentParser(description="DIT TextBlock inference script")
    parser.add_argument(
        "--image_path",
        help="Path to input image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        help="Name of the output visualization directory.",
        type=str,
    )
    parser.add_argument(
        "--config-file",
        default="configs/mask_rcnn_dit_base.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

    
    
