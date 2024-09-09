import os
from typing import List, Tuple
from queue import Queue
from threading import Lock, Event

from api.common.typedef import Infer
import cv2
import tensorflow as tf


def image_to_tensor(path):
    img = cv2.imread(path)  # type: ignore
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # type: ignore
    tensor = tf.expand_dims(tf.convert_to_tensor(rgb, dtype=tf.uint8), 0)
    return tensor


class Detector:

    def __init__(self, saved_model_path: str, threshold: float):
        # saved model from savedmodel_path
        self.detect_fn = tf.saved_model.load(saved_model_path)
        self.thold = threshold
        self.frames_queue: Queue[Tuple[Event, List[str]]] = Queue()
        self.frames_queue_lock = Lock()
        self.inference_queue: Queue[List[List[Infer]]] = Queue()

    def run_inference(self, img_path: str) -> List[Infer]:

        # convert to tensor and run inference
        input_tensor = image_to_tensor(img_path)  # type: ignore
        detections = self.detect_fn(input_tensor)  # type: ignore

        res = []
        # draw bounding boxes whose scores are larger than threshold
        boxes = detections["detection_boxes"][0].numpy()
        scores = detections["detection_scores"][0].numpy()

        name, _ = os.path.splitext(img_path)
        detections_txt_path = f"{name}-detection"
        detection_txt_file = open(detections_txt_path, "w")
        for i in range(boxes.shape[0]):
            if scores[i] >= self.thold:
                res.append(boxes[i])
                detection_txt_file.write(  # type: ignore
                        f"{boxes[i][0]} {boxes[i][1]} "
                        f"{boxes[i][2]} {boxes[i][3]}\n"
                        )
        detection_txt_file.close()

        return res

    def run(self):
        while True:
            done_flag: Event  # type: ignore
            frame_paths: List[str]
            while self.frames_queue.empty():
                pass
            done_flag, frame_paths = self.frames_queue.get()
            inferences: List[List[Infer]] = [self.run_inference(frame)
                                             for frame in frame_paths]
            self.inference_queue.put(inferences)
            done_flag.set()
