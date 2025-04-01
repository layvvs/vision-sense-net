from .net import ONNXNet
import numpy as np
from dataclasses import dataclass


YOLO_INPUT_SIZE = 640
CONFIDENSE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4


@dataclass
class Facebox:
    left: int
    top: int
    right: int
    bottom: int
    confidence: float

    @property
    def box_ltrb(self):
        return (self.left, self.top, self.right, self.bottom)


class YOLO(ONNXNet):
    model_name = 'yolo-face'

    def iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def nms(self, boxes, confidences, iou_threshold):
        if len(boxes) == 0:
            return [], []

        sorted_indices = np.argsort(confidences)[::-1]
        keep_boxes = []

        while len(sorted_indices) > 0:
            best_idx = sorted_indices[0]
            best_box = boxes[best_idx]
            keep_boxes.append(best_idx)

            remaining_indices = sorted_indices[1:]
            filtered_indices = []

            for idx in remaining_indices:
                if self.iou(best_box, boxes[idx]) < iou_threshold:
                    filtered_indices.append(idx)

            sorted_indices = np.array(filtered_indices)

        return boxes[keep_boxes], confidences[keep_boxes]

    def postprocess(
        self,
        outputs,
        orig_w,
        orig_h,
        conf_threshold=CONFIDENSE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD
    ):
        squeezed_output = np.squeeze(outputs)  # (5, 8400)

        boxes = squeezed_output[:4, :].T  # (8400, 4)
        confidences = squeezed_output[4, :]  # (8400,)

        x_center = boxes[:, 0]
        y_center = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]

        left = x_center - width / 2
        top = y_center - height / 2
        right = x_center + width / 2
        bottom = y_center + height / 2

        left = (left / YOLO_INPUT_SIZE) * orig_w
        top = (top / YOLO_INPUT_SIZE) * orig_h
        right = (right / YOLO_INPUT_SIZE) * orig_w
        bottom = (bottom / YOLO_INPUT_SIZE) * orig_h

        boxes = np.stack([left, top, right, bottom], axis=1)

        mask = confidences > conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]

        boxes, confidences = self.nms(boxes, confidences, iou_threshold)

        return boxes, confidences

    def preprocess(self, frame):
        resized_frame = frame.resize((YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
        processed_frame = np.array(resized_frame).transpose(2, 0, 1).astype(np.float32) / 255
        return np.expand_dims(processed_frame, axis=0)

    def run(self, frame) -> list[Facebox]:
        processed_frame = self.preprocess(frame.body)
        net_result = self._run(processed_frame)
        boxes, confidences = self.postprocess(net_result, frame.width, frame.height)
        return [
            Facebox(box[0], box[1], box[2], box[3], confidense)
            for box, confidense in zip(boxes, confidences)
        ]
