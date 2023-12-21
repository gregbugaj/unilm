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


def process_dir(predictor: DefaultPredictor, image_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    for idx, img_path in enumerate(glob.glob(os.path.join(image_dir, "*.*"))):
        if not img_path.endswith((".jpg", ".png", ".jpeg", ".tif", ".tiff")):
            continue
        try:
            print(img_path)
            filename = os.path.splitext(os.path.basename(img_path) )[0]
            output_file_name = os.path.join(output_dir, f"{filename}.png")

            inference(predictor, img_path, output_file_name)
        except Exception as e:
            print(e)
            # raise e

def inference(predictor:DefaultPredictor, image_path: str, output_path: str):
    print(f"Inference on image: {image_path}")
    img = cv2.imread(image_path)
    output = predictor(img)["instances"]
    

    # md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    # if cfg.DATASETS.TEST[0]=='icdar2019_test':
    #     md.set(thing_classes=["table"])
    # else:
    #     md.set(thing_classes=["text","title","list","table","figure"])

    # create empty metadata catalog
    from detectron2.data import MetadataCatalog
    md = MetadataCatalog.get("ditod_test")
    md.set(thing_classes=["cropped_page","header","footer","sidebar"])


    v = Visualizer(img[:, :, ::-1],
                    md,
                    scale=1.0,
                    instance_mode=ColorMode.SEGMENTATION)
    
    result = v.draw_instance_predictions(output.to("cpu"))
    result_image = result.get_image()[:, :, ::-1]
    
    cv2.imwrite(output_path, result_image)


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
    
    args.image_path = os.path.expanduser(args.image_path)
    args.output_path = os.path.expanduser(args.output_path)
    # check if image path is a directory or a file
    # Step 5: run inference
    print(f"Starting inference on {args.image_path}")
    print(f"Output path: {args.output_path}")

    if os.path.isdir(args.image_path):
        print("Image path is a directory")
        process_dir(predictor, args.image_path, args.output_path)
    else:
        print("Image path is a file")
        inference(predictor, args.image_path, args.output_path)
    
    # cv2.imwrite(args.output_file_name, result_image)


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

