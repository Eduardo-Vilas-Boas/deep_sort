import numpy
import torch
from torchvision.ops import nms
from ultralytics import YOLO

from deep_sort.detection import Detection


class ImageObjectDetector(object):
    _instance = None
    object_detector_model = None
    image_encoder = None
    min_confidence = None
    min_height = None
    nms_max_overlap = None

    def __new__(
        cls,
        model_path,
        image_encoder,
        min_confidence,
        min_height,
        nms_max_overlap,
    ):
        if not isinstance(cls._instance, cls):
            cls._instance = super(ImageObjectDetector, cls).__new__(cls)
            cls._instance.load_model(
                model_path,
                image_encoder,
                min_confidence,
                min_height,
                nms_max_overlap,
            )
        return cls._instance

    @classmethod
    def load_model(
        cls,
        model_path,
        image_encoder,
        min_confidence,
        min_height,
        nms_max_overlap,
    ):
        cls.object_detector_model = YOLO(model_path)
        cls.image_encoder = image_encoder
        cls.min_confidence = min_confidence
        cls.min_height = min_height
        cls.nms_max_overlap = nms_max_overlap

    @classmethod
    def update_model(cls, model_path, image_encoder):
        cls._instance.load_model(model_path, image_encoder)

    def __call__(self, image):

        results = self.object_detector_model(image, verbose=False)

        image = image.transpose(2, 1, 0)
        detections = []

        for result in results:

            conf_list = result.boxes.conf
            bbox_list = result.boxes.xywh
            label_list = result.boxes.cls

            # Apply NMS
            keep = nms(bbox_list, conf_list, self.nms_max_overlap)

            # Keep only the boxes that were not suppressed
            conf_list = conf_list[keep].tolist()
            label_list = label_list[keep].tolist()
            bbox_list = bbox_list[keep].tolist()

            for conf, _, bbox in zip(conf_list, label_list, bbox_list):
                if bbox[3] < self.min_height:
                    continue
                if conf < self.min_confidence:
                    continue

                # print("bbox:", bbox)

                image_crop = image[
                    :,
                    int(bbox[0]) : int(bbox[0] + bbox[2]),
                    int(bbox[1]) : int(bbox[1] + bbox[3]),
                ]

                image_crop = numpy.expand_dims(image_crop, axis=0)
                # print("image_crop:", image_crop.shape)

                feature = self.image_encoder(image_crop).cpu().detach().numpy()

                detections.append(Detection(bbox, conf, feature))
        return detections
