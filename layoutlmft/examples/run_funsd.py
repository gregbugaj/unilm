#!/usr/bin/env python
# coding=utf-8

from gc import callbacks
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image, ImageDraw, ImageFont

import numpy as np
from datasets import ClassLabel, load_dataset, load_metric
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D

import transformers
from layoutlmft.data import DataCollatorForKeyValueExtraction
from layoutlmft.data.data_args import DataTrainingArguments
from layoutlmft.models.model_args import ModelArguments
from layoutlmft.trainers import FunsdTrainer as Trainer
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    set_seed,
    LayoutLMv2Config
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

from transformers import LayoutLMv2Processor, LayoutLMv2FeatureExtractor, LayoutLMv2ForTokenClassification, \
    LayoutLMv2TokenizerFast

import multiprocessing as mp

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers
os.environ['TRANSFORMERS_CACHE'] = '/data/cache/'

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")

logger = logging.getLogger(__name__)

feature_size = 224  # 224
batch_size   = 32 #  28
 
##Next, let's use `LayoutLMv2Processor` to prepare the data for the model.
# 115003 / 627003

# feature_extractor = LayoutLMv2FeatureExtractor(size = 672, apply_ocr=False)
feature_extractor = LayoutLMv2FeatureExtractor(size = feature_size, apply_ocr=False)
tokenizer = LayoutLMv2TokenizerFast.from_pretrained("microsoft/layoutlmv2-large-uncased")# microsoft/layoutlmv2-base-uncased
processor = LayoutLMv2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

from transformers import TrainerCallback

class ModelTrackingCallback(TrainerCallback):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_metric = 1000

    def on_train_begin(self, args, state, control, **kwargs):
        print(f"Starting training : {state.is_local_process_zero}")


    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)

        if state.is_local_process_zero:
            print(logs)

    def on_save(self, args, state, control, **kwargs):
        print(f"model saved ---- : {state.best_metric}, {self.best_metric },  {state.is_local_process_zero}")
        
        print(state)
        if state.is_local_process_zero:
            import os
            from pathlib import Path
            import shutil, errno

            def copy_and_overwrite(from_path, to_path):
                if os.path.exists(to_path):
                    shutil.rmtree(to_path)
                shutil.copytree(from_path, to_path)
                
            best_dir = os.path.join(args.output_dir, "best")

            if state.best_metric < self.best_metric:
                self.best_metric = state.best_metric
                print(f"Saving best [{self.best_metric }]: {state.best_model_checkpoint}")
                os.makedirs(best_dir, exist_ok=True)
                try:
                    copy_and_overwrite(state.best_model_checkpoint, best_dir)                
                except Exception as ex:
                    print(ex)
                

    # TrainerState(epoch=0.11160714285714286, global_step=50, max_steps=22400, num_train_epochs=50, total_flos=967199925731328.0, 
    # log_history=[{'loss': 1.4162, 'learning_rate': 4.991960696739616e-05, 'epoch': 0.11, 'step': 50},
    #  {'eval_loss': 0.34052422642707825, 'eval_precision': 0.675246152877207, 'eval_recall': 0.7972832283972039, 'eval_f1': 0.7312077294685991, 'eval_accuracy': 0.9114714962790795, 
    #  'eval_runtime': 29.2985, 'eval_samples_per_second': 40.787, 'eval_steps_per_second': 10.205, 'epoch': 0.11, 'step': 50}], 
    #  best_metric=0.34052422642707825, best_model_checkpoint='/home/gbugaj/dev/unilm/layoutlmft/examples/checkpoints-tuned-pan/checkpoint-50', 
    #  is_local_process_zero=False, is_world_process_zero=False, is_hyper_param_search=False, trial_name=None, trial_params=None)




            
def main():
    # See all possible arguments in layoutlmft/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # datasets = load_dataset(os.path.abspath(layoutlmft.data.datasets.funsd.__file__))

    # datasets = load_dataset(os.path.abspath(layoutlmft.funsd_dataset.__file__))
    datasets = load_dataset("funsd_dataset/funsd_dataset.py", cache_dir="/data/cache/")

    print(datasets)
    labels = datasets['train'].features['ner_tags'].feature.names

    print('NER-Labels -> ')
    print(labels)

    id2label = {v: k for v, k in enumerate(labels)}
    label2id = {k: v for v, k in enumerate(labels)}

    print("ID2Label : ")
    print(id2label)
    print(label2id) 

    # os.exit()


    if training_args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    else:
        column_names = datasets["validation"].column_names
        features = datasets["validation"].features
    text_column_name = "tokens" if "tokens" in column_names else column_names[0]
    label_column_name = (
        f"{data_args.task_name}_tags" if f"{data_args.task_name}_tags" in column_names else column_names[1]
    )

    remove_columns = column_names

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
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = LayoutLMv2Config.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,

        hidden_dropout_prob = .5,
        attention_probs_dropout_prob = .5,
    )


    # model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutlmv2-base-uncased', num_labels=len(labels))
    # model = LayoutLMv2ForTokenClassification.from_pretrained("./checkpoints", num_labels=len(labels))

    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    #     cache_dir=model_args.cache_dir,
    #     use_fast=True,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    # )

    
    model = LayoutLMv2ForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # print(tokenizer)


    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )


    print('Tokenzier type : ')
    print(type(tokenizer))

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):

        print("-----------")
        print(text_column_name)
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            return_overflowing_tokens=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        labels = []
        bboxes = []
        images = []
        for batch_index in range(len(tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][batch_index]

            label = examples[label_column_name][org_batch_index]
            bbox = examples["bboxes"][org_batch_index]
            image = examples["image"][org_batch_index]
            previous_word_idx = None
            label_ids = []
            bbox_inputs = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                    bbox_inputs.append([0, 0, 0, 0])
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                    bbox_inputs.append(bbox[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label_to_id[label[word_idx]] if data_args.label_all_tokens else -100)
                    bbox_inputs.append(bbox[word_idx])
                previous_word_idx = word_idx
            labels.append(label_ids)
            bboxes.append(bbox_inputs)
            images.append(image)
        tokenized_inputs["labels"] = labels
        tokenized_inputs["bbox"] = bboxes
        tokenized_inputs["image"] = images
        return tokenized_inputs


    # we need to define custom features
    features = Features({
        'image': Array3D(dtype="int64", shape=(3, feature_size, feature_size)), # 224
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int64')),
        'token_type_ids': Sequence(Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        'labels': Sequence(ClassLabel(names=labels)),
    })

    def preprocess_data(examples):
        images = [Image.open(path).convert("RGB") for path in examples['image_path']]
        # images = [image for image in examples['image']]
        words = examples['words']
        boxes = examples['bboxes']
        word_labels = examples['ner_tags']
        
        encoded_inputs = processor(images, words, boxes=boxes, word_labels=word_labels, padding="max_length", truncation=True)

        return encoded_inputs

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

        processes = 4 # int(mp.cpu_count() // 4)
        train_dataset = train_dataset.map(preprocess_data, batched=True, remove_columns=train_dataset.column_names,
                                    features=features, num_proc = processes)

        # train_dataset = train_dataset.map(
        #     tokenize_and_align_labels,
        #     batched=True,
        #     remove_columns=remove_columns,
        #     num_proc=data_args.preprocessing_num_workers,
        #     load_from_cache_file=not data_args.overwrite_cache,
        # )

    if training_args.do_eval:
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
            
        processes = 4 # int(mp.cpu_count() // 4)
        eval_dataset = eval_dataset.map(preprocess_data, batched=True, remove_columns=eval_dataset.column_names,
                                    features=features, num_proc = processes)

        # eval_dataset = eval_dataset.map(
        #     tokenize_and_align_labels,
        #     batched=True,
        #     remove_columns=remove_columns,
        #     num_proc=data_args.preprocessing_num_workers,
        #     load_from_cache_file=not data_args.overwrite_cache,
        # )

    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

        processes = 4 # int(mp.cpu_count() // 4)
        test_dataset = datasets['test'].map(preprocess_data, batched=True, remove_columns=datasets['test'].column_names,
                                      features=features, num_proc = processes)

        # test_dataset = test_dataset.map(
        #     tokenize_and_align_labels,
        #     batched=True,
        #     remove_columns=remove_columns,
        #     num_proc=data_args.preprocessing_num_workers,
        #     load_from_cache_file=not data_args.overwrite_cache,
        # )

    # Data collator
    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        padding=padding,
        max_length=512,
    )

    # Metrics
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
        if data_args.return_entity_level_metrics:
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

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks = [ModelTrackingCallback]
    )


    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(test_dataset)
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        # Save predictions
        output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_test_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()
