import argparse
import glob
import logging
import os
import sys
import torch
import cv2

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from docarray import DocumentArray
from torch.nn import Module
from ditod import add_vit_config

from detectron2.utils.visualizer import ColorMode, Visualizer

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.engine import DefaultPredictor

from ditod import MyTrainer, add_vit_config

import argparse

logger = logging.getLogger(__name__)

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
    # device = "cpu"
    cfg.MODEL.DEVICE = device

    cfg.freeze()
    # default_setup(cfg, args)
    return cfg

def optimize_model(model):
    """Optimizes the model for inference. This method is called by the __init__ method."""
    try:
        import torchvision.models as models
        import torch._dynamo as dynamo

        # ['aot_eager', 'aot_eager_decomp_partition', 'aot_torchxla_trace_once', 'aot_torchxla_trivial', 'aot_ts', 'aot_ts_nvfuser', 'cudagraphs', 'dynamo_accuracy_minifier_backend', 'dynamo_minifier_backend', 'eager', 'inductor', 'ipex', 'nvprims_aten', 'nvprims_nvfuser', 'onnxrt', 'torchxla_trace_once', 'torchxla_trivial', 'ts', 'tvm']
        torch._dynamo.config.verbose = False
        torch._dynamo.config.suppress_errors = True
        # torch.backends.cudnn.benchmark = True
        # https://dev-discuss.pytorch.org/t/torchinductor-update-4-cpu-backend-started-to-show-promising-performance-boost/874
        # ipex
        model = torch.compile(model)
           
        # model = torch.compile(model, backend="onnxrt", fullgraph=False)
        # model = torch.compile(model)
        return model
    except Exception as err:
        logger.warning(f"Model compile not supported: {err}")
        raise err


def build_model_from_config(cfg):
    """
    Returns:
        torch.nn.Module:

    It now calls :func:`detectron2.modeling.build_model`.
    Overwrite it if you'd like a different model.
    """
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    return optimize_model(model)
    # return model


def process_dir(predictor: DefaultPredictor, image_dir: str):
    for idx, img_path in enumerate(glob.glob(os.path.join(image_dir, "*.*"))):
        if not img_path.endswith((".jpg", ".png", ".jpeg", ".tif", ".tiff")):
            continue

        try:
            print(img_path)
            inference(predictor, img_path)
        except Exception as e:
            print(e)
            # raise e

def inference(predictor:DefaultPredictor, image_path: str):

    img = cv2.imread(image_path)
    output = predictor(img)["instances"]
    md=None

    # print(output)

    v = Visualizer(img[:, :, ::-1],
                    md,
                    scale=1.0,
                    instance_mode=ColorMode.SEGMENTATION)
    result = v.draw_instance_predictions(output.to("cpu"))
    result_image = result.get_image()[:, :, ::-1]


    filename = os.path.basename(image_path)    
    filename = os.path.splitext(filename)[0]

    output_filename = f"/tmp/dit/result_{filename}.png"
    print(output_filename)
    cv2.imwrite(output_filename, result_image)


def main(args):
    print("Inference")
    # Step 1: instantiate config
    cfg = setup_cfg(args)
    print(cfg)
    model = build_model_from_config(cfg)    
    # Step 4: define model
    predictor = DefaultPredictor(cfg)

    if args.image_path is None:
        print("No image path provided")
        return
    
    # check if image path is a directory or a file
    # Step 5: run inference

    if os.path.isdir(args.image_path):
        print("Image path is a directory")
        process_dir(predictor, args.image_path)
    else:
        print("Image path is a file")
        inference(predictor, args.image_path)
    
    # cv2.imwrite(args.output_file_name, result_image)


### Sample usage
###  python ./inference.py --config-file configs/mask_rcnn_dit_base.yaml  --image_path /home/gbugaj/tmp/2022-08-09/PID_1549_8460_0_159274481.tif --output_path /tmp/dit --opts  MODEL.WEIGHTS /home/gbugaj/models/unilm/dit/text_detection/td-syn_dit-b_mrcnn.pth

### python ./inference.py --config-file configs/mask_rcnn_dit_large.yaml  --image_path /home/greg/tmp/marie-cleaner/to-clean-001/burst/00001.tif --output_path /tmp/dit --opts  MODEL.WEIGHTS /mnt/data/marie-ai/model_zoo/unilm/dit/text_detection/td-syn_dit-l_mrcnn.pth

### python ./inference.py --config-file configs/mask_rcnn_dit_large.yaml  --image_path /home/greg/datasets/private/medprov/PID/150300411/burst  --output_path /tmp/dit --opts  MODEL.WEIGHTS /mnt/data/marie-ai/model_zoo/unilm/dit/text_detection/td-syn_dit-l_mrcnn.pth


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

    
    