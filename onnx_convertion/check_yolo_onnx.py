import onnxruntime
from PIL import Image, ImageDraw
from torchvision import transforms
import numpy as np


prep = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])


model = onnxruntime.InferenceSession('/Users/layvvs/Desktop/Studying/BMSTU/Diploma-2024-2025/code/vision-sense-net/models/yolov8n-face.onnx')
img = Image.open('/Users/layvvs/Desktop/Studying/BMSTU/Diploma-2024-2025/code/vision-sense-net/onnx_convertion/2025-01-13 01.23.49.jpg')
img_prep = prep(img)
res = model.run(None, {'images': np.expand_dims(img_prep.numpy(), axis=0)})


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def nms(boxes, confidences, iou_threshold=0.4):
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
            if iou(best_box, boxes[idx]) < iou_threshold:
                filtered_indices.append(idx)

        sorted_indices = np.array(filtered_indices)

    return boxes[keep_boxes], confidences[keep_boxes]


def postprocess_output(outputs, orig_w, orig_h, conf_threshold=0.5, iou_threshold=0.4):
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

    left = (left / 640) * orig_w
    top = (top / 640) * orig_h
    right = (right / 640) * orig_w
    bottom = (bottom / 640) * orig_h

    boxes = np.stack([left, top, right, bottom], axis=1)

    mask = confidences > conf_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]

    boxes, confidences = nms(boxes, confidences, iou_threshold)

    return boxes, confidences

boxes, confidences = postprocess_output(res, img.width, img.height)


def draw_boxes(img: Image, boxes, confidences):
    draw = ImageDraw.Draw(img)

    for box, conf in zip(boxes, confidences):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), f"{conf:.2f}", fill="blue")

    img.show()


draw_boxes(img, boxes, confidences)
