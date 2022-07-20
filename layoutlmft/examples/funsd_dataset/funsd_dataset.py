import json
import os
from functools import lru_cache

import datasets
import ujson
import orjson

import multiprocessing as mp
from concurrent.futures.thread import ThreadPoolExecutor

import concurrent.futures
import time
from multiprocessing import Pool

from PIL import Image, ImageDraw, ImageFont
import numpy as np

import torch

logger = datasets.logging.get_logger(__name__)

_CITATION = """ N/A """
_DESCRIPTION = """ N/A """


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)


def load_imageXXX(image_path):
    from detectron2.data.detection_utils import read_image
    from detectron2.data.transforms import ResizeTransform, TransformList

    image = read_image(image_path, format="BGR")
    h = image.shape[0]
    w = image.shape[1]
    img_trans = TransformList([ResizeTransform(h=h, w=w, new_h=224, new_w=224, interp=Image.LANCZOS)])
    image = torch.tensor(img_trans.apply_image(image).copy()).permute(2, 0, 1)  # copy to make it writeable
    return image, (w, h)

# @lru_cache(maxsize=20)
@lru_cache()
def normalize_bbox(bbox, size):
    res = [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]
    # print(f"  >  {bbox}  ->  {res}")
    return res

class FunsdConfig(datasets.BuilderConfig):
    """BuilderConfig for FUNSDLike"""

    def __init__(self, **kwargs):
        """BuilderConfig for FUNSDLike.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(FunsdConfig, self).__init__(**kwargs)

class FunsdLikeDataset(datasets.GeneratorBasedBuilder):
    """FUNSD LIKE dataset for NER extraction"""

    BUILDER_CONFIGS = [
        FunsdConfig(name="funsd_dataset", version=datasets.Version("1.6.0"), description="FUNSD Like dataset"),
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
                            names = [
                             "O",
                             'B-MEMBER_NAME', 'I-MEMBER_NAME',
                             'B-MEMBER_NAME_ANSWER', 'I-MEMBER_NAME_ANSWER',
                             'B-MEMBER_NUMBER', 'I-MEMBER_NUMBER', 
                             'B-MEMBER_NUMBER_ANSWER', 'I-MEMBER_NUMBER_ANSWER', 
                             'B-PAN', 'I-PAN', 'B-PAN_ANSWER', 'I-PAN_ANSWER', 'B-DOS', 'I-DOS', 'B-DOS_ANSWER', 'I-DOS_ANSWER', 
                             'B-PATIENT_NAME', 'I-PATIENT_NAME', 'B-PATIENT_NAME_ANSWER', 'I-PATIENT_NAME_ANSWER',
                             'B-HEADER', 'I-HEADER',
                             'B-DOCUMENT_CONTROL', 'I-DOCUMENT_CONTROL',
                             'B-LETTER_DATE', 'I-LETTER_DATE',
                             'B-PARAGRAPH', 'I-PARAGRAPH',
                             'B-ADDRESS', 'I-ADDRESS',
                             'B-QUESTION', 'I-QUESTION',
                             'B-ANSWER', 'I-ANSWER',
                             'B-PHONE', 'I-PHONE',
                             'B-URL', 'I-URL',
                             'B-GREETING', 'I-GREETING',
                             ]
                        )
                    ),
                    "image_path": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager): #-> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        # downloaded_file = dl_manager.download_and_extract("https://guillaumejaume.github.io/FUNSD/dataset.zip")
        # downloaded_file = "/home/greg/dataset/assets-private/corr-indexer"
        # downloaded_file = "/home/greg/dataset/assets-private/corr-indexer-converted"                          
        # downloaded_file = "/home/greg/dataset/funsd"
        downloaded_file = "/home/gbugaj/dataset/private/corr-indexer-converted"
        downloaded_file = "/data/dataset/private/corr-indexer-augmented"
        downloaded_file = "/home/greg/dataset/assets-private/corr-indexer-augmented"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": f"{downloaded_file}/dataset/training_data/"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": f"{downloaded_file}/dataset/testing_data/"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": f"{downloaded_file}/dataset/validation_data/"}
            ),
        ]

    def _generate_(self, filepath, guid, file):
        ann_dir = os.path.join(filepath, "annotations")
        img_dir = os.path.join(filepath, "images")

        t_start = time.time()
        # print(f'\n guid = {guid},  {t_start},  {filepath}')

        file_path = os.path.join(ann_dir, file)
        with open(file_path, "r", encoding="utf8",  buffering=1024*1024) as f:
            _ = f.read() 
            data = orjson.loads(_)
            # data = json.load(f)
            # data = ujson.load(f)
        
        t_end = time.time()
        # print(f't_end = {t_end},  {t_end - t_start}')
            
        words = []
        bboxes = []
        ner_tags = []            
        bboxes_raw = []
        
        image_path = os.path.join(img_dir, file)
        image_path = image_path.replace("json", "png")
        image, size = load_image(image_path)

        word_index = 0
        font = ImageFont.load_default()

        for item in data["form"]:
            words_example, label = item["words"], item["label"]
            # print(words_example)
            # remap bad 'text:' label with `:`                
            for w in words_example:
                if "text:" in w:
                    w["text"] = w["text:"]

            words_example = [w for w in words_example if w["text"].strip() != ""]

            if len(words_example) == 0:
                continue
            if label == "other":
                for w in words_example:
                    words.append(w["text"])
                    ner_tags.append("O")
                    bboxes.append(normalize_bbox(tuple(w["box"]), size))
                    bboxes_raw.append(w["box"])
            else:
                # label = _label
                words.append(words_example[0]["text"])
                ner_tags.append("B-" + label.upper())
                bboxes.append(normalize_bbox(tuple(words_example[0]["box"]), size))

                bboxes_raw.append(words_example[0]["box"])

                for w in words_example[1:]:
                    words.append(w["text"])
                    ner_tags.append("I-" + label.upper())
                    bboxes.append(normalize_bbox(tuple(w["box"]), size))
                    bboxes_raw.append(w["box"])

            words_lower = [w.lower() for w in words]
            # words_upper = [w.upper() for w in words]
            # print ({"id": str(guid), "words": words, "bboxes": bboxes, "ner_tags": ner_tags, "image_path": image_path})
            # draw predictions over the image
            # print(f"{word_index} => {words_lower}")

        if False:
            draw = ImageDraw.Draw(image)

            for tag, box in zip(ner_tags, bboxes_raw):
                text = f"{word_index}"
                draw.rectangle(box, outline='red')
                draw.text((box[0] + 10, box[1] - 10), text=text, fill='red' )
                # print(f"\t\t\t {word_index} => {tag} -> {words_lower}")
                word_index +=1
            image.save(f"/tmp/snippet/{guid}.png")
        
        t_end = time.time()
        # print(f'f_end = {t_end}')
        # yield guid, {"id": str(guid), "words": words_upper, "bboxes": bboxes, "ner_tags": ner_tags, "image_path": image_path}
        return  guid, {"id": str(guid), "words": words_lower, "bboxes": bboxes, "ner_tags": ner_tags, "image_path": image_path}         

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "annotations")
        img_dir = os.path.join(filepath, "images")
        font = ImageFont.load_default()

        args = []
        items = sorted(os.listdir(ann_dir))
        np.random.shuffle(items)
        stop = int(len(items) *.1)
        stop = int(len(items))
        
        for guid, file in enumerate(items):
            file_path = os.path.join(ann_dir, file)
            __args = (filepath, guid, file)
            args.append(__args)
            if guid == stop:
                break

        results = []
        start = time.time()
        print("\nPool Executor:")
        print("Time elapsed: %s" % (time.time() - start))

        processes = int(mp.cpu_count() * .95)
        # processes = 1
        pool = Pool(processes=processes)
        pool_results = pool.starmap(self._generate_, args)

        pool.close()
        pool.join()

        print("Time elapsed[submitted]: %s" % (time.time() - start))
        for r in pool_results:
            # print("Time elapsed[result]: %s  , %s" % (time.time() - start, r))
            yield r

        print("Time elapsed[all]: %s" % (time.time() - start))
