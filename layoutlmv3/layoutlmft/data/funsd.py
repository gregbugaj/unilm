# coding=utf-8
'''
Reference: https://huggingface.co/datasets/nielsr/funsd/blob/main/funsd.py
'''
import json
import os
from re import L

from PIL import Image

import datasets
import multiprocessing as mp
from concurrent.futures.thread import ThreadPoolExecutor
import concurrent.futures
import time
from multiprocessing import Pool

from PIL import Image, ImageDraw, ImageFont
import numpy as np

import torch

def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)

def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]

logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{Jaume2019FUNSDAD,
  title={FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents},
  author={Guillaume Jaume and H. K. Ekenel and J. Thiran},
  journal={2019 International Conference on Document Analysis and Recognition Workshops (ICDARW)},
  year={2019},
  volume={2},
  pages={1-6}
}
"""

_DESCRIPTION = """\
https://guillaumejaume.github.io/FUNSD/
"""


class FunsdConfig(datasets.BuilderConfig):
    """BuilderConfig for FUNSD"""

    def __init__(self, **kwargs):
        """BuilderConfig for FUNSD.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(FunsdConfig, self).__init__(**kwargs)


class Funsd(datasets.GeneratorBasedBuilder):
    """Conll2003 dataset."""

    BUILDER_CONFIGS = [
        FunsdConfig(name="funsd", version=datasets.Version("1.8.0"), description="FUNSD like dataset, corr-indexing"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    # "ner_tags": datasets.Sequence(
                    #     datasets.features.ClassLabel(
                    #         names=["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
                    #     )
                    # ),

                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                'B-MEMBER_NAME', 'I-MEMBER_NAME',
                                'B-MEMBER_NUMBER', 'I-MEMBER_NUMBER',
                                'B-PAN', 'I-PAN',
                                'B-PATIENT_NAME', 'I-PATIENT_NAME',
                                'B-DOS', 'I-DOS',
                                'B-DOS_ANSWER', 'I-DOS_ANSWER',
                                'B-PATIENT_NAME_ANSWER', 'I-PATIENT_NAME_ANSWER',
                                'B-MEMBER_NAME_ANSWER', 'I-MEMBER_NAME_ANSWER',
                                'B-MEMBER_NUMBER_ANSWER', 'I-MEMBER_NUMBER_ANSWER',
                                'B-PAN_ANSWER', 'I-PAN_ANSWER',
                                'B-ADDRESS', 'I-ADDRESS',
                                'B-GREETING', 'I-GREETING',
                                'B-HEADER', 'I-HEADER',
                                'B-LETTER_DATE', 'I-LETTER_DATE',
                                'B-PARAGRAPH', 'I-PARAGRAPH',
                                'B-QUESTION', 'I-QUESTION',
                                'B-ANSWER', 'I-ANSWER',
                                'B-DOCUMENT_CONTROL', 'I-DOCUMENT_CONTROL',
                                'B-PHONE', 'I-PHONE',
                                'B-URL', 'I-URL',
                                'B-CLAIM_NUMBER', 'I-CLAIM_NUMBER',
                                'B-CLAIM_NUMBER_ANSWER', 'I-CLAIM_NUMBER_ANSWER',
                                'B-BIRTHDATE', 'I-BIRTHDATE',
                                'B-BIRTHDATE_ANSWER', 'I-BIRTHDATE_ANSWER',
                                'B-BILLED_AMT', 'I-BILLED_AMT',
                                'B-BILLED_AMT_ANSWER', 'I-BILLED_AMT_ANSWER',
                                'B-PAID_AMT', 'I-PAID_AMT',
                                'B-PAID_AMT_ANSWER', 'I-PAID_AMT_ANSWER',
                                'B-CHECK_AMT', 'I-CHECK_AMT',
                                'B-CHECK_AMT_ANSWER', 'I-CHECK_AMT_ANSWER',
                                'B-CHECK_NUMBER', 'I-CHECK_NUMBER',
                                'B-CHECK_NUMBER_ANSWER', 'I-CHECK_NUMBER_ANSWER',
                                'B-LIST', 'I-LIST',
                                'B-FOOTER', 'I-FOOTER',
                                'B-DATE', 'I-DATE',
                                'B-IDENTIFIER', 'I-IDENTIFIER',
                                'B-PROC_CODE', 'I-PROC_CODE',
                                'B-PROC_CODE_ANSWER', 'I-PROC_CODE_ANSWER',
                                'B-PROVIDER', 'I-PROVIDER',
                                'B-PROVIDER_ANSWER', 'I-PROVIDER_ANSWER',
                                'B-MONEY', 'I-MONEY',
                                'B-COMPANY', 'I-COMPANY',
                            ]
                        )
                    ),

                    "image": datasets.features.Image(),
                    # "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    "image_path": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://guillaumejaume.github.io/FUNSD/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # downloaded_file = dl_manager.download_and_extract("https://guillaumejaume.github.io/FUNSD/dataset.zip")

        downloaded_file = "/data/dataset/private/corr-indexer-augmented"
        # downloaded_file = "/home/greg/dataset/assets-private/corr-indexer-augmented"
        downloaded_file = "/home/gbugaj/datasets/private/corr-indexer-augmented"
        downloaded_file = "/home/greg/datasets/private/assets-private/corr-indexer-augmented"
        # downloaded_file = "/home/gbugaj/dataset/private/corr-indexer-augmented"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": f"{downloaded_file}/dataset/training_data/"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": f"{downloaded_file}/dataset/testing_data/"}
            ),
        ]


    def get_line_bbox(self, bboxs):
        x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
        y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

        assert x1 >= x0 and y1 >= y0
        bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
        return bbox

    def _generate_(self, guid, file):
        logger.info("⏳ Generating examples from = %s", self.filepath)
        ann_dir = os.path.join(self.filepath, "annotations")
        img_dir = os.path.join(self.filepath, "images")

        tokens = []
        bboxes = []
        ner_tags = []


        file_path = os.path.join(ann_dir, file)
        with open(file_path, "r", encoding="utf8") as f:
            data = json.load(f)
        image_path = os.path.join(img_dir, file)
        image_path = image_path.replace("json", "png")
        image, size = load_image(image_path)

        # somehow we got a TEXT token with size of [0,0,W,H]
        # TODO: investigate
        # example :/corr-indexer-augmented/dataset/training_data/annotations/152658473_2_140_0.json
        #       "words": [
        # {
        #   "box": [
        #     0,
        #     0,
        #     776,
        #     1000
        #   ],
        #   "text": ":"
        # }
        # ]
    
        for item in data["form"]:
            cur_line_bboxes = []
            words, label = item["words"], item["label"]
            # remap bad 'text:' label with `:`
            for w in words:
                if "text:" in w:
                    w["text"] = w["text:"]

            words = [w for w in words if w["text"].strip() != ""]
            if len(words) == 0:
                continue

            if label == "other":
                for w in words:
                    # TODO: How did we endup with O-Token with size of [0,0,W,H]
                    other_box  = normalize_bbox(w["box"], size)
                    if other_box[0]== 0 and other_box[1] == 0:
                        continue

                    tokens.append(w["text"])
                    ner_tags.append("O")
                    cur_line_bboxes.append(other_box)
            else:
                tokens.append(words[0]["text"])
                ner_tags.append("B-" + label.upper())
                cur_line_bboxes.append(normalize_bbox(words[0]["box"], size))
                for w in words[1:]:
                    tokens.append(w["text"])
                    ner_tags.append("I-" + label.upper())
                    cur_line_bboxes.append(normalize_bbox(w["box"], size))

            # by default: --segment_level_layout 1
            # if do not want to use segment_level_layout, comment the following line
            if len(cur_line_bboxes) == 0:
                print(f"Empty cur_line_bboxes for {words} : {file_path}")
                continue
                
            cur_line_bboxes = self.get_line_bbox(cur_line_bboxes)
            bboxes.extend(cur_line_bboxes)


        if len(bboxes) == 0:
            payload = {"id": str(guid), "tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags}
            print(f"Empty Boxes for : {file_path}")
            return 

        return guid, {"id": str(guid), "tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags,
                        "image": image, "image_path": image_path}


    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "annotations")
        img_dir = os.path.join(filepath, "images")

        args = []
        items = sorted(os.listdir(ann_dir))
        np.random.shuffle(items)

        stop = int(len(items) *.50)
        stop = int(len(items))
        
        # https://stackoverflow.com/questions/47776486/python-struct-error-i-format-requires-2147483648-number-2147483647
        self.filepath = filepath
        for guid, file in enumerate(items):
            file_path = os.path.join(ann_dir, file)
            __args = (guid, file)
            args.append(__args)
            if guid == stop:
                break

        results = []
        start = time.time()
        print("\nPool Executor:")
        print("Time elapsed: %s" % (time.time() - start))

        processes = int(mp.cpu_count() * .90)
        processes = 4
        pool = Pool(processes=processes)
        pool_results = pool.starmap(self._generate_, args)

        pool.close()
        pool.join()

        print("Time elapsed[submitted]: %s" % (time.time() - start))
        for r in pool_results:
            # print("Time elapsed[result]: %s  , %s" % (time.time() - start, r))
            yield r

        print("Time elapsed[all]: %s" % (time.time() - start))                         