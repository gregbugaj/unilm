#!/usr/bin/env python
# coding=utf-8
from PIL import Image, ImageDraw, ImageFont

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

# https://github.com/huggingface/transformers/blob/d3ae2bd3cf9fc1c3c9c9279a8bae740d1fd74f34/tests/layoutlmv2/test_processor_layoutlmv2.py


# Calling this from here prevents : "AttributeError: module 'detectron2' has no attribute 'config'"
from detectron2.config import get_cfg

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")
logger = logging.getLogger(__name__)


from datasets import load_dataset
# Calling this from here prevents : "AttributeError: module 'detectron2' has no attribute 'config'"
# from detectron2.config import get_cfg

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


def main_dataset():
    datasets = load_dataset("nielsr/funsd")
    print(datasets)

    labels = datasets['train'].features['ner_tags'].feature.names
    print("labels == ")    
    # print(labels)['O', 'B-HEADER', 'I-HEADER', 'B-QUESTION', 'I-QUESTION', 'B-ANSWER', 'I-ANSWER']

    id2label = {v: k for v, k in enumerate(labels)}
    label2id = {k: v for v, k in enumerate(labels)}

    #    Let's test the trained model on the first image of the test set:

    example = datasets["test"][0]
    print(example.keys())

    image = Image.open(example['image_path'])
    image = image.convert("RGB")
    image.show()

    # print("example['ner_tags']")
    # print(example['words'])
    # print(example['bboxes'])

    processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

    encoded_inputs = processor(image, example['words'], boxes=example['bboxes'], word_labels=example['ner_tags'],
                               padding="max_length", truncation=True, return_tensors="pt")

    # Next, let's move everything to the GPU, if it's available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    labels = encoded_inputs.pop('labels').squeeze().tolist()
    for k, v in encoded_inputs.items():
        encoded_inputs[k] = v.to(device)

    # load the fine-tuned model from the hub
    model = LayoutLMv2ForTokenClassification.from_pretrained("nielsr/layoutlmv2-finetuned-funsd")
    # model = torch.load("./tuned/layoutlmv2-finetuned-funsd-torch.pth")
    model.to(device)

    import time
    start_time = time.time()
    model.eval()
    outputs = model(**encoded_inputs)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(outputs.logits.shape)

    # Let's create the true predictions, true labels (in terms of label names) as well as the true boxes.

    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoded_inputs.bbox.squeeze().tolist()

    width, height = image.size

    true_predictions = [id2label[prediction] for prediction, label in zip(predictions, labels) if label != -100]
    true_labels = [id2label[label] for prediction, label in zip(predictions, labels) if label != -100]
    true_boxes = [unnormalize_box(box, width, height) for box, label in zip(token_boxes, labels) if label != -100]

    print(true_predictions)
    print(true_labels)

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    label2color = {'question': 'blue', 'answer': 'green', 'header': 'orange', 'other': 'violet'}

    for prediction, box in zip(true_predictions, true_boxes):
        predicted_label = iob_to_label(prediction).lower()
        draw.rectangle(box, outline=label2color[predicted_label])
        draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label], font=font)

    image.show()


def main_image():

    labels = ['O', 'B-HEADER', 'I-HEADER', 'B-QUESTION', 'I-QUESTION', 'B-ANSWER', 'I-ANSWER']
    logger.info("Labels : {}", labels)

    id2label = {v: k for v, k in enumerate(labels)}
    label2id = {k: v for v, k in enumerate(labels)}

    # prepare for the model
    # we do not want to use the pytesseract 
    # LayoutLMv2FeatureExtractor requires the PyTesseract library but it was not found in your environment. You can install it with pip:
    processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
    
    # Create Layout processor with custom future extractor
    # feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
    # processor = LayoutLMv2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # image = Image.open('/home/gbugaj/data/eval/funsd/__results___28_0.png').convert("RGB")
    image = Image.open('./document.png').convert("RGB")
    # image.show()

    # Next, let's move everything to the GPU, if it's available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## Need to obtain boxes/ for the document
    words = ["weirdly", "world"]
    boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]
    word_labels = [1, 2]

    encoded_inputs = processor(image, words, boxes=boxes, word_labels=word_labels, return_tensors="pt")
    expected_keys = ["attention_mask", "bbox", "image", "input_ids", "token_type_ids"]
    actual_keys = sorted(list(encoded_inputs.keys()))

    print("Expected Keys : ", expected_keys)
    print("Actual Keys : ", actual_keys)

    for key in expected_keys:
        print(encoded_inputs[key])


    labels = encoded_inputs.pop('labels').squeeze().tolist()
    for k, v in encoded_inputs.items():
        encoded_inputs[k] = v.to(device)

    print(labels)

    # load the fine-tuned model from the hub
    model = LayoutLMv2ForTokenClassification.from_pretrained("nielsr/layoutlmv2-finetuned-funsd")
    model.to(device)

    # forward pass
    outputs = model(**encoded_inputs)
    print(outputs.logits.shape)

    # Let's create the true predictions, true labels (in terms of label names) as well as the true boxes.

    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoded_inputs.bbox.squeeze().tolist()

    width, height = image.size

    true_predictions = [id2label[prediction] for prediction, label in zip(predictions, labels) if label != -100]
    true_labels = [id2label[label] for prediction, label in zip(predictions, labels) if label != -100]
    true_boxes = [unnormalize_box(box, width, height) for box, label in zip(token_boxes, labels) if label != -100]

    print(true_predictions)
    print(true_labels)
    print(true_boxes)


    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    def iob_to_label(label):
        label = label[2:]
        if not label:
            return 'other'
        return label

    label2color = {'question': 'blue', 'answer': 'green', 'header': 'orange', 'other': 'violet'}

    for prediction, box in zip(true_predictions, true_boxes):
        predicted_label = iob_to_label(prediction).lower()
        draw.rectangle(box, outline=label2color[predicted_label])
        draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label], font=font)

    image.show()


if __name__ == "__main__":
    main_dataset()
    main_image()

