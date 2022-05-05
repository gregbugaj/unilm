import json
import os

import datasets

from PIL import Image
import numpy as np


import torch

from detectron2.data.detection_utils import read_image
from detectron2.data.transforms import ResizeTransform, TransformList


logger = datasets.logging.get_logger(__name__)

_CITATION = """ N/A """
_DESCRIPTION = """ N/A """


def __scale_height(img, target_size, method=Image.LANCZOS):
    ow, oh = img.size
    scale = oh / target_size
    print(scale)
    w = ow / scale
    h = target_size  # int(max(oh / scale, crop_size))
    return img.resize((int(w), int(h)), method)


def load_imageXXX(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)


def load_image(image_path):
    image = read_image(image_path, format="BGR")
    h = image.shape[0]
    w = image.shape[1]
    img_trans = TransformList([ResizeTransform(h=h, w=w, new_h=224, new_w=224, interp=Image.LANCZOS)])
    image = torch.tensor(img_trans.apply_image(image).copy()).permute(2, 0, 1)  # copy to make it writeable
    return image, (w, h)


def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]

class FunsdConfig(datasets.BuilderConfig):
    """BuilderConfig for FUNSD"""

    def __init__(self, **kwargs):
        """BuilderConfig for FUNSD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(FunsdConfig, self).__init__(**kwargs)

class FunsdLikeDataset(datasets.GeneratorBasedBuilder):
    """FUNSD LIKE dataset."""

    BUILDER_CONFIGS = [
        FunsdConfig(name="funsd_dataset", version=datasets.Version("1.0.0"), description="FUNSD Like dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "words": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            # names=["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
                            names = ["O", 'B-MEMBER_NAME', 'I-MEMBER_NAME', 'B-MEMBER_NAME_ANSWER', 'I-MEMBER_NAME_ANSWER', 'B-MEMBER_NUMBER', 'I-MEMBER_NUMBER', 'B-MEMBER_NUMBER_ANSWER', 'I-MEMBER_NUMBER_ANSWER', 'B-PAN', 'I-PAN', 'B-PAN_ANSWER', 'I-PAN_ANSWER', 'B-DOS', 'I-DOS', 'B-DOS_ANSWER', 'I-DOS_ANSWER', 'B-PATIENT_NAME', 'I-PATIENT_NAME', 'B-PATIENT_NAME_ANSWER', 'I-PATIENT_NAME_ANSWER']
                        )
                    ),
                    "image_path": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # downloaded_file = dl_manager.download_and_extract("https://guillaumejaume.github.io/FUNSD/dataset.zip")
        downloaded_file = "/home/greg/dataset/assets-private/corr-indexer"
        # downloaded_file = "/home/gbugaj/dataset/private/corr-indexer"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": f"{downloaded_file}/dataset/train_dataset/"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": f"{downloaded_file}/dataset/test_dataset/"}
            ),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "annotations")
        img_dir = os.path.join(filepath, "images")
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            words = []
            bboxes = []
            ner_tags = []
            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, file)
            image_path = image_path.replace("json", "png")
            image, size = load_image(image_path)
            for item in data["form"]:
                words_example, label = item["words"], item["label"]
                words_example = [w for w in words_example if w["text"].strip() != ""]
                if len(words_example) == 0:
                    continue
                if label == "other":
                    for w in words_example:
                        words.append(w["text"])
                        ner_tags.append("O")
                        bboxes.append(normalize_bbox(w["box"], size))
                else:
                    words.append(words_example[0]["text"])
                    ner_tags.append("B-" + label.upper())
                    bboxes.append(normalize_bbox(words_example[0]["box"], size))
                    for w in words_example[1:]:
                        words.append(w["text"])
                        ner_tags.append("I-" + label.upper())
                        bboxes.append(normalize_bbox(w["box"], size))
            yield guid, {"id": str(guid), "words": words, "bboxes": bboxes, "ner_tags": ner_tags, "image_path": image_path}
            