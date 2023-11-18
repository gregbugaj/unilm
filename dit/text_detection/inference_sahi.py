# will be used for detectron2 fasterrcnn model zoo name
from sahi.utils.detectron2 import Detectron2TestConstants

# import required functions, classes
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict, get_prediction
from sahi.utils.file import download_from_url
from sahi.utils.cv import read_image
from IPython.display import Image

from detectron2sahi import Detectron2DitDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi.utils.cv import visualize_object_predictions


# detection_model = AutoDetectionModel.from_pretrained(
#     model_type='detectron2',

#     confidence_threshold=0.5,
#     image_size=768,
#     device="cuda:0", # or 'cuda:0'
# )




detection_model = Detectron2DitDetectionModel(
    model_path="/home/gbugaj/models/eob-extract/model_0026999.pth",
    config_path="/home/gbugaj/dev/unilm/dit/text_detection/configs/mask_rcnn_dit_base.yaml",
    device='cuda' # or 'cpu'
)


result = get_sliced_prediction(
    # "/home/gbugaj/datasets/private/eob-extract/converted/imgs/eob-extract/eob-002/159185251_3.png",
    "/home/gbugaj/datasets/private/eob-extract/converted/imgs/eob-extract/eob-002/159016456_5.png",
    # "/home/gbugaj/datasets/private/eob-extract/scaled-159185251_3.png",
    detection_model,
    slice_height = 786,
    slice_width = 786,
    overlap_height_ratio = 0.45,
    overlap_width_ratio = 0.45,
)

result.export_visuals(export_dir="/tmp/dit")
