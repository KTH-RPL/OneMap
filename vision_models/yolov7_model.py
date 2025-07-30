import sys

import cv2
import numpy as np
import torch
import time
from vision_models.utils.yolov7_utils import letterbox, non_max_suppression, scale_coords, check_img_size
from vision_models.coco_classes import COCO_CLASSES
# a = sys.path.pop(0)
sys.path.insert(0, "yolov7/")
#try:
from models.experimental import attempt_load  # noqa: E402
from utils.datasets import letterbox  # noqa: E402
from utils.general import (  # noqa: E402
    check_img_size,
    non_max_suppression,
    scale_coords,
)
from utils.torch_utils import TracedModel  # noqa: E402
#except Exception:
#    print("Could not import yolov7. This is OK if you are only using the client.")
sys.path.pop(0)
# sys.path.insert(0, a)
class YOLOv7Detector:
    def __init__(self,
                 confidence_threshold: float
                 ):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.half_precision = self.device.type != "cpu"
        self.model = attempt_load('weights/yolov7-e6e.pt', map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.image_size = check_img_size(640, s=stride)  # check img_size
        self.model = TracedModel(self.model, self.device, self.image_size)
        if self.half_precision:
            self.model.half()  # to
        self.confidence_threshold = confidence_threshold
        self.classes = None
        self.class_map = {}
        # Translate 'tv_monitor', 'sofa', 'chair', 'bed', 'toilet'
        self.class_map["chair"] = "chair"
        self.class_map["tv"] = "tv_monitor"
        self.class_map["potted plant"] = "plant"
        self.class_map["couch"] = "sofa"
        self.class_map["bed"] = "bed"
        self.class_map["toilet"] = "toilet"
        classes_oi = ["chair", "tv", "potted plant", "bed", "toilet", "couch"]
        self.classes_oi = [COCO_CLASSES.index(c) for c in classes_oi]
        # self.classes_oi = None

    def predict(self, image):
        results = self.model(image)
        return results

    def set_classes(self,
                    classes: list
                    ):
        self.classes = classes

    def detect(self,
               image: np.ndarray
               ):
        a = time.time()
        orig_shape = image.shape
        # NOTE removed this, as it was not used in the original code. 
        # NOTE We added this to experiment with VLFM's yolov7 setup when investigating false positives.
        # img = cv2.resize(
        #      image,
        #      (self.image_size, int(self.image_size * 0.7)),
        #      interpolation=cv2.INTER_AREA,
        # )
        img = image
        img = letterbox(img, new_shape=self.image_size)[0]
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to("cuda")

        img = img.half() if self.half_precision else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.inference_mode():  # Calculating gradients causes a GPU memory leak
            pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred,
            self.confidence_threshold,
            0.45,
            classes=self.classes_oi,
            agnostic=False,
        )[0]
        # Rescale boxes from img_size to im0 size
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], orig_shape).round()
        pred[:, 0] = torch.clip(pred[:, 0], 0, orig_shape[1] - 1)
        pred[:, 1] = torch.clip(pred[:, 1], 0, orig_shape[0] - 1)
        pred[:, 2] = torch.clip(pred[:, 2], 0, orig_shape[1] - 1)
        pred[:, 3] = torch.clip(pred[:, 3], 0, orig_shape[0] - 1)
        boxes = pred[:, :4]
        logits = pred[:, 4]
        preds = {}
        preds["boxes"] = []
        preds["scores"] = []
        for i in range(pred.shape[0]):
            class_name = COCO_CLASSES[int(pred[i, 5])]
            if class_name == self.classes[0]:
                box = boxes[i]
                if not (box[0].item() == box[2].item() or box[1].item() == box[3].item()):
                    preds["boxes"].append([box[0].item(), box[1].item(), box[2].item(), box[3].item()])
                    preds["scores"].append(logits[i])
                    # print(logits[i])

        # print(f"YOLO forward: {time.time() - a}")
        return preds

if __name__ == "__main__":
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # clip = ClipModel('weights/clip.pth')
    detector = YOLOv7Detector(0.5)
    detector.set_classes(["couch"])
    image = cv2.imread("/home/spot/Pictures/sofa.jpeg")
    image = cv2.resize(image, (640, 640))
    # print(image.shape)
    preds = detector.detect(image)
    start.record()
    preds = detector.detect(image)
    end.record()
    torch.cuda.synchronize()
    print("Complete forward: ", start.elapsed_time(end) / 1000)
    print(preds)
    print("Done!")
