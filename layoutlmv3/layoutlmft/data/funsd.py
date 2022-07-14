# coding=utf-8
'''
Reference: https://huggingface.co/datasets/nielsr/funsd/blob/main/funsd.py
'''
import json
import os

import datasets

from layoutlmft.data.image_utils import load_image, normalize_bbox


logger = datasets.logging.get_logger(__name__)


_CITATION = """\

"""

_DESCRIPTION = """\
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
    """dataset."""

    BUILDER_CONFIGS = [
        FunsdConfig(name="funsd", version=datasets.Version("1.9.0"), description="FUNSD dataset"),
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
                            names = [
                             "O", 
                             'B-MEMBER_NAME', 'I-MEMBER_NAME', 
                             'B-MEMBER_NAME_ANSWER', 'I-MEMBER_NAME_ANSWER',
                             'B-MEMBER_NUMBER', 'I-MEMBER_NUMBER', 
                             'B-MEMBER_NUMBER_ANSWER', 'I-MEMBER_NUMBER_ANSWER', 
                             'B-PAN', 'I-PAN', 
                             'B-PAN_ANSWER', 'I-PAN_ANSWER', 
                             'B-DOS', 'I-DOS', 
                             'B-DOS_ANSWER', 'I-DOS_ANSWER', 
                             'B-PATIENT_NAME', 'I-PATIENT_NAME', 
                             'B-PATIENT_NAME_ANSWER', 'I-PATIENT_NAME_ANSWER',
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
                    
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
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

        downloaded_file = "/home/greg/dataset/assets-private/corr-indexer-converted"   
        downloaded_file = "/data/dataset/private/corr-indexer-augmented"
        # downloaded_file = "/home/greg/dataset/assets-private/corr-indexer-converted"   
        downloaded_file = "/home/greg/dataset/assets-private/corr-indexer-augmented"   

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

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "annotations")
        img_dir = os.path.join(filepath, "images")
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            tokens = []
            bboxes = []
            ner_tags = []

            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, file)
            image_path = image_path.replace("json", "png")
            image, size = load_image(image_path)
            for item in data["form"]:
                

                cur_line_bboxes = []
                words, label = item["words"], item["label"]

                # remap bad 'text:' label with `:`                
                for w in words:
                    if "text:" in w:
                        w["text"] = w["text:"]

                for w in words :
                    if "text" not in w:
                        print(w)
                        raise Exception("EX")

                words = [w for w in words if w["text"].strip() != ""]


                if len(words) == 0:
                    continue
                if label == "other":
                    for w in words:
                        tokens.append(w["text"])
                        ner_tags.append("O")
                        cur_line_bboxes.append(normalize_bbox(w["box"], size))
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
                cur_line_bboxes = self.get_line_bbox(cur_line_bboxes)
                # box = normalize_bbox(item["box"], size)
                # cur_line_bboxes = [box for _ in range(len(words))]
                bboxes.extend(cur_line_bboxes)
            yield guid, {"id": str(guid), "tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags,
                         "image": image, "image_path": image_path}
