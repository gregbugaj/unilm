#!/usr/bin/env python
# --------------------------------------------------------------------------------
# MPViT: Multi-Path Vision Transformer for Dense Prediction
# Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI).
# All Rights Reserved.
# Written by Youngwan Lee
# --------------------------------------------------------------------------------

"""
Detection Training Script for MPViT.
"""
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data.datasets import register_coco_instances

from ditod import MyTrainer, add_vit_config

from distutils.util import strtobool as strtobool
import warnings as _warnings
import os as _os

if strtobool(_os.environ.get("SUPPRESS_WARNINGS", "true")):
    # attempt to suppress all warnings from dependencies

    _warnings.simplefilter(action="ignore", category=FutureWarning)
    _warnings.simplefilter(action="ignore", category=UserWarning)
    _warnings.simplefilter(action="ignore", category=DeprecationWarning)

    # # Compiled functions can't take variable number of arguments or use keyword-only arguments with defaults
    try:

        def warn(*args, **kwargs):
            pass

        # _warnings.warn = warn

        # Work around for https://github.com/pytorch/pytorch/issues/29637
        # We will owerwrite the formatting function for warnings to make it not print anything
    except Exception as ex:
        pass
else:

    def _warning_on_one_line(message, category, filename, lineno, *args, **kwargs):
        return "\033[1;33m%s: %s\033[0m \033[1;30m(raised from %s:%s)\033[0m\n" % (
            category.__name__,
            message,
            filename,
            lineno,
        )

    _warnings.formatwarning = _warning_on_one_line
    _warnings.simplefilter("always", DeprecationWarning)



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # add_coat_config(cfg)
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.DATALOADER.NUM_WORKERS = 1  # Torch crashes if we use > 1 workers
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    register_coco_instances(
        "funsd_train",
        {},
        "data/instances_training.json",
        # "data/imgs"
        "data/"# Slice dataset
    )

    register_coco_instances(
        "funsd_test",
        {},
        "data/instances_test.json",
        # "data/imgs"
        "data/"
    )

    cfg = setup(args)

    if args.eval_only:
        model = MyTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = MyTrainer.test(cfg, model)
        return res

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    args = parser.parse_args()
    print("Command Line Args:", args)
    
    if args.debug:
        import debugpy

        print("Enabling attach starts.")
        debugpy.listen(address=('0.0.0.0', 9310))
        debugpy.wait_for_client()
        print("Enabling attach ends.")

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
