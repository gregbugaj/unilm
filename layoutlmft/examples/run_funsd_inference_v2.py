#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch

import numpy as np
from transformers.utils import check_min_version

from PIL import Image
from transformers import LayoutLMv2Processor, LayoutLMv2FeatureExtractor, LayoutLMv2ForTokenClassification
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D

# https://github.com/huggingface/transformers/blob/d3ae2bd3cf9fc1c3c9c9279a8bae740d1fd74f34/tests/layoutlmv2/test_processor_layoutlmv2.py


from datasets import load_dataset
# Calling this from here prevents : "AttributeError: module 'detectron2' has no attribute 'config'"
from detectron2.config import get_cfg
from PIL import Image, ImageDraw, ImageFont

datasets = load_dataset("nielsr/funsd")
print(datasets)

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
# processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
model = LayoutLMv2ForTokenClassification.from_pretrained("nielsr/layoutlmv2-finetuned-funsd")

# load image example
dataset = load_dataset("nielsr/funsd", split="test")
image = Image.open(dataset[0]["image_path"]).convert("RGB")
image.save("document.png")

# define id2label, label2color
labels = dataset.features['ner_tags'].feature.names
id2label = {v: k for v, k in enumerate(labels)}
label2color = {'question': 'blue', 'answer': 'green', 'header': 'orange', 'other': 'violet'}


def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]


def iob_to_label(label):
    label = label[2:]
    if not label:
        return 'other'
    return label


def process_image(image):
    width, height = image.size

    # encode
    encoding = processor(image, truncation=True, return_offsets_mapping=True, return_tensors="pt")
    offset_mapping = encoding.pop('offset_mapping')

    # forward pass
    outputs = model(**encoding)

    # get predictions
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()

    # only keep non-subword predictions
    is_subword = np.array(offset_mapping.squeeze().tolist())[:, 0] != 0
    true_predictions = [id2label[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
    true_boxes = [unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]

    # draw predictions over the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for prediction, box in zip(true_predictions, true_boxes):
        predicted_label = iob_to_label(prediction).lower()
        draw.rectangle(box, outline=label2color[predicted_label])
        draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label], font=font)

    return image


def main():
    # Will error if the minimal version of Transformers is not installed. Remove at your own risks.
    check_min_version("4.5.0")
    logger = logging.getLogger(__name__)

    resolved = process_image(image)
    resolved.show()


def mainVV():
    # Will error if the minimal version of Transformers is not installed. Remove at your own risks.
    check_min_version("4.5.0")

    logger = logging.getLogger(__name__)

    # prepare for the model
    # we do not want to use the pytesseract 
    # LayoutLMv2FeatureExtractor requires the PyTesseract library but it was not found in your environment. You can install it with pip:
    processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

    feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
    # processor = LayoutLMv2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # image = Image.open('/home/gbugaj/data/eval/funsd/__results___28_0.png').convert("RGB")
    image = Image.open('/home/gbugaj/data/training/funsd/dataset/testing_data/images/82092117.png').convert("RGB")
    words = ["weirdly", "world"]
    boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]
    word_labels = [1, 2]
    input_processor = processor(image, words, boxes=boxes, word_labels=word_labels, return_tensors="pt")
    expected_keys = ["attention_mask", "bbox", "image", "input_ids", "token_type_ids"]
    actual_keys = sorted(list(input_processor.keys()))

    print(actual_keys)
    for key in expected_keys:
        print(input_processor[key])


if __name__ == "__main__":
    main()
