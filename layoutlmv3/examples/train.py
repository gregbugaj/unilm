import os
import random

import numpy as np
import torch
import transformers
from PIL import Image, ImageDraw, ImageFont

from torch.utils.data import DataLoader
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor, AdamW

from transformers import LayoutLMv3Processor, LayoutLMv3FeatureExtractor, LayoutLMv3ForTokenClassification, \
    LayoutLMv3TokenizerFast,    AutoConfig


from tqdm import tqdm

# Calling this from here prevents : "AttributeError: module 'detectron2' has no attribute 'config'"
from detectron2.config import get_cfg

from datasets import load_dataset, load_metric
from datasets.features import ClassLabel
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D

import multiprocessing as mp
from transformers import get_scheduler
from distutils.version import LooseVersion

import warnings
warnings.filterwarnings('ignore')

# this dataset uses the new Image feature :)
from transformers import AutoProcessor
from transformers import AutoModelForTokenClassification
from transformers import LayoutLMv3ForTokenClassification

from transformers import TrainingArguments, Trainer
from transformers import TrainingArguments
from transformers.data.data_collator import default_data_collator


use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
print(device)
device_ids = [0]



os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers
# os.environ['TRANSFORMERS_CACHE'] = '/tmp/cache/'
print(transformers.__version__)

dataset = load_dataset("funsd_dataset/funsd_dataset.py", cache_dir="/data/cache")

# print(dataset['train'].features)
print(dataset['train'].features['bboxes'])

labels = dataset['train'].features['ner_tags'].feature.names

print('NER-Labels -> ')
print(labels)

id2label = {v: k for v, k in enumerate(labels)}
label2id = {k: v for v, k in enumerate(labels)}

print("ID2Label : ")
print(id2label)
print(label2id)


if False:
    example = dataset["train"][0]
    # example["image"].show()
    words, boxes, ner_tags = example["tokens"], example["bboxes"], example["ner_tags"]
    print(words)
    print(boxes)
    print(ner_tags)

# we'll use the Auto API here - it will load LayoutLMv3Processor behind the scenes,
# based on the checkpoint we provide from the hub
# processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

model_name_or_path = "microsoft/layoutlmv3-large"
model_name_or_path = "microsoft/layoutlmv3-base"

config = AutoConfig.from_pretrained (
    model_name_or_path,
    num_labels=len(labels),
    finetuning_task="ner",
    cache_dir="/mnt/data/cache",
    input_size=224,
    id2label=id2label,
    label2id=label2id,
    hidden_dropout_prob = .2,
    attention_probs_dropout_prob = .2,
    # has_relative_attention_bias=False
)

# Max model size is 512, so we will need to handle any documents larger thank that
feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False, do_resize=True, resample=Image.LANCZOS) # BICUBIC
tokenizer = LayoutLMv3TokenizerFast.from_pretrained(model_name_or_path, is_split_into_words=True)
processor = LayoutLMv3Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

features = dataset["train"].features
column_names = dataset["train"].column_names
image_column_name = "image"
text_column_name = "tokens"
boxes_column_name = "bboxes"
label_column_name = "ner_tags"

# In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
# unique labels.
def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list

if isinstance(features[label_column_name].feature, ClassLabel):
    label_list = features[label_column_name].feature.names
    # No need to convert the labels since they are already ints.
    id2label = {k: v for k,v in enumerate(label_list)}
    label2id = {v: k for k,v in enumerate(label_list)}
else:
    label_list = get_label_list(dataset["train"][label_column_name])
    id2label = {k: v for k,v in enumerate(label_list)}
    label2id = {v: k for k,v in enumerate(label_list)}
    
num_labels = len(label_list)


# print(label_list)
# print(id2label)

def prepare_examples (examples):
  images = examples[image_column_name]
  words = examples[text_column_name]
  boxes = examples[boxes_column_name]
  word_labels = examples[label_column_name]

#   encoding = processor(images, words, boxes=boxes, word_labels=word_labels, truncation=True,  padding="max_length")

  encoding = processor(images, words, boxes=boxes, word_labels=word_labels, truncation=True, stride = 128, 
         padding="max_length", max_length=512, return_overflowing_tokens=True, return_offsets_mapping=True,
  )  
  
  offset_mapping = encoding.pop('offset_mapping')
  overflow_to_sample_mapping = encoding.pop('overflow_to_sample_mapping')
  
  return encoding


def prepare_examplesXX(examples):
    images = examples[image_column_name]
    words = examples[text_column_name]
    boxes = examples[boxes_column_name]
    word_labels = examples[label_column_name]

    #   encoding = processor(images, words, boxes=boxes, word_labels=word_labels, truncation=True,  padding="max_length")

    #   encoding = processor(images, words, boxes=boxes, word_labels=word_labels, truncation=True, stride = 128, 
    #          padding="max_length", max_length=512, return_overflowing_tokens=True, return_offsets_mapping=True)  

    #   offset_mapping = encoding.pop('offset_mapping')
    #   overflow_to_sample_mapping = encoding.pop('overflow_to_sample_mapping')

    stride = 128

    tokenized_inputs = processor(
        images,
        words,
        boxes= boxes,
        word_labels=word_labels,
        # is_split_into_words=True,
        add_special_tokens=True,
        truncation=True,
        stride=stride,
        padding=True,
        max_length=512,
        return_overflowing_tokens=True,  # important !!!
        return_length=True,
        verbose=True,
    )

    # print(tokenized_inputs)
    
    tag_everything = False


    labels = []

    i = 0
    label = examples["ner_tags"]
    overflowing = tokenized_inputs[i].overflowing

    print("len overflowing: ", len(overflowing))
    print("len label: ", len(label))
    print("len tokenized_inputs A: ", len(tokenized_inputs["labels"]))

    for i in range(0, 1 + len(overflowing)):
        word_ids = tokenized_inputs[i].word_ids

        print("len word_ids: ", len(word_ids))
        print(word_ids)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(np.zeros_like(label[0]))
            elif (word_idx != previous_word_idx) or (tag_everything):
                label_ids.append(label[word_idx])
            else:
                label_ids.append(np.zeros_like(label[0]))
            previous_word_idx = word_idx

        print("len label_ids: ", len(label_ids))
        labels.extend(label_ids)

    # tokenized_inputs["labels"] = labels

    print("len tokenized_inputs X: ", len(tokenized_inputs["labels"]))

    return tokenized_inputs


# we need to define custom features for `set_format` (used later on) to work properly
features = Features({
    'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'labels': Sequence(feature=Value(dtype='int64')),
})

# We
# dataset['train'] = dataset['train'].shuffle(seed=42)
# dataset['test'] = dataset['test'].shuffle(seed=42)

train_dataset = dataset["train"].map(
    prepare_examples,
    batched=True,
    remove_columns=column_names,
    features=features,
    num_proc = 4
)

eval_dataset = dataset["test"].map(
    prepare_examples,
    batched=True,
    remove_columns=column_names,
    features=features,
    num_proc = 4
)


example = train_dataset[0]
processor.tokenizer.decode(example["input_ids"])
train_dataset.set_format("torch")


example = train_dataset[0]
for k,v in example.items():
    print(k,v.shape)

# for id, label in zip(train_dataset[0]["input_ids"], train_dataset[0]["labels"]):
#   print(processor.tokenizer.decode([id]), label.item())


return_entity_level_metrics = False
metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


# Define the model
model = LayoutLMv3ForTokenClassification.from_pretrained(
    model_name_or_path,
    config=config,
)

training_argsXXX = TrainingArguments(
                num_train_epochs = 25,
                max_steps=50000,
                save_steps = 2000,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=12,
                learning_rate=4e-5,
                evaluation_strategy="steps",
                eval_steps=2000,
                load_best_model_at_end=True,                  
                greater_is_better=True, 
                do_train=True, 
                do_eval=True,
                metric_for_best_model="f1",
                output_dir="/data/models/layoutlmv3-base-20230711-stride128",
                # resume_from_checkpoint="/data/models/layoutlmv3-large-20230711-stride128/checkpoint-1000/",
                fp16=True,
            )

training_args = TrainingArguments(
                num_train_epochs = 50,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                learning_rate=2e-5,
                weight_decay=0.01,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,                  
                greater_is_better=True, 
                metric_for_best_model="f1",
                output_dir="/data/models/layoutlmv3-base-20230711-stride128",
                # resume_from_checkpoint="/data/models/layoutlmv3-large-20230711-stride128/checkpoint-1000/",
                fp16=True,
            )


print('training_args *************************')
print(training_args)

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

if True:
    # checkpoint = last_checkpoint if last_checkpoint else None
    train_result = trainer.train()

    metrics = train_result.metrics
    trainer.save_model()  # Saves the tokenizer too for easy upload

    # max_train_samples = (
    #     data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    # )
    # metrics["train_samples"] = min(max_train_samples, len(train_dataset))
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    trainer.evaluate()

# TODO : how to handle this:
# Token indices sequence length is longer than the specified maximum sequence length for this model (541 > 512). Running this sequence through the model will result in indexing errors

os.exit()
# Inference
model_name_or_path = "/mnt/data/models/layoutlmv3-large-finetuned-funsd/checkpoint-5000"
model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)


# example = dataset["test"][random.randint(0, len(dataset["test"]))]
index = random.randrange(0, len(dataset["test"]), 1)
print(index)

# Token indices sequence length is longer than the specified maximum sequence length for this model (585 > 512). Running this sequence through the model will result in indexing errors
# 129
example = dataset["test"][111]
print(example.keys())

image = example["image"]
words = example["tokens"]
boxes = example["bboxes"]
word_labels = example["ner_tags"]

encoding = processor(image, words, boxes=boxes, word_labels=word_labels, truncation=True, padding="max_length", return_tensors="pt")

# encoding = processor(image, words, boxes=boxes, word_labels=word_labels, padding="max_length", return_tensors="pt")
for k,v in encoding.items():
  print(k,v.shape)

with torch.no_grad():
  outputs = model(**encoding)

logits = outputs.logits
predictions = logits.argmax(-1).squeeze().tolist()
print(predictions)

labels = encoding.labels.squeeze().tolist()
print(labels)

def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]


token_boxes = encoding.bbox.squeeze().tolist()
width, height = image.size

true_predictions = [model.config.id2label[pred] for pred, label in zip(predictions, labels) if label != - 100]
true_labels = [model.config.id2label[label] for prediction, label in zip(predictions, labels) if label != -100]
true_boxes = [unnormalize_box(box, width, height) for box, label in zip(token_boxes, labels) if label != -100]


from PIL import ImageDraw, ImageFont

draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

def iob_to_label(label):
    label = label[2:]
    if not label:
      return 'other'
    return label

label2color = {'question':'blue', 'answer':'green', 'header':'orange', 'other':'violet'}
label2color = {
        "pan": "blue",
        "pan_answer": "green",
        "dos": "orange",
        "dos_answer": "violet",
        "member": "blue",
        "member_answer": "green",
        "member_number": "blue",
        "member_number_answer": "green",
        "member_name": "blue",
        "member_name_answer": "green",
        "patient_name": "blue",
        "patient_name_answer": "green",
        "paragraph": "purple",
        "greeting": "blue",
        "address": "orange",
        "question": "blue",
        "answer": "aqua",
        "document_control": "grey",
        "header": "brown",
        "letter_date": "deeppink",
        "url": "darkorange",
        "phone": "darkmagenta",
        "other": "red",

        "claim_number": "darkmagenta",
        "claim_number_answer": "green",
        "birthdate": "green",
        "birthdate_answer": "red",
        "billed_amt": "green",
        "billed_amt_answer": "orange",
        "paid_amt": "green",
        "paid_amt_answer": "blue",
        "check_amt": "orange",
        "check_amt_answer": "darkmagenta",
        "check_number": "orange",
        "check_number_answer": "blue",
    }

for prediction, box in zip(true_predictions, true_boxes):
    predicted_label = iob_to_label(prediction).lower()
    draw.rectangle(box, outline=label2color[predicted_label])
    draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label], font=font)

image.save("/tmp/tensors/prediction.png")

image = example["image"]
image = image.convert("RGB").copy()

draw = ImageDraw.Draw(image)

for word, box, label in zip(example['tokens'], example['bboxes'], example['ner_tags']):
  actual_label = iob_to_label(id2label[label]).lower()
  box = unnormalize_box(box, width, height)
  draw.rectangle(box, outline=label2color[actual_label], width=2)
  draw.text((box[0] + 10, box[1] - 10), actual_label, fill=label2color[actual_label], font=font)

image.save("/tmp/tensors/real.png")



# REF :
# https://github.com/rossumai/docile/blob/ffc139e8e37505121c4b49243011ceed18653650/baselines/NER/docile_train_NER_multilabel_layoutLMv3.py#L23