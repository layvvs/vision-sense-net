from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import onnxruntime as rt
import numpy as np


CLASSES = ('angry', 'happy', 'neutral', 'sad', 'surprize')
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
YOLO_INPUT_SIZE = 640


prep = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


def softmax(x):
    x_exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return x_exp / np.sum(x_exp, axis=-1, keepdims=True)


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

    left = (left / YOLO_INPUT_SIZE) * orig_w
    top = (top / YOLO_INPUT_SIZE) * orig_h
    right = (right / YOLO_INPUT_SIZE) * orig_w
    bottom = (bottom / YOLO_INPUT_SIZE) * orig_h

    boxes = np.stack([left, top, right, bottom], axis=1)

    mask = confidences > conf_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]

    boxes, confidences = nms(boxes, confidences, iou_threshold)

    return boxes, confidences


yolo = rt.InferenceSession('/Users/layvvs/Desktop/Studying/BMSTU/Diploma-2024-2025/code/vision-sense-net/models/yolov8n-face.onnx')

img = Image.open('image.png')

img_yolo = np.expand_dims(np.array(img.resize((640, 640))).transpose(2, 0, 1).astype(np.float32) / 255, axis=0)

yolo_results = yolo.run(None, {'images': np.array(img_yolo)})

boxes, confidences = postprocess_output(yolo_results, img.width, img.height)

for box, conf in zip(boxes, confidences):

    img_croped = img.crop(box.squeeze())
    img_resnet = np.expand_dims(prep(img_croped).numpy(), axis=0)
    resnet = rt.InferenceSession('/Users/layvvs/Desktop/Studying/BMSTU/Diploma-2024-2025/code/vision-sense-net/models/fer_resnet50.onnx')

    resnet_results = resnet.run(None, {'images': img_resnet})

    print(CLASSES[np.argmax(softmax(resnet_results))])

    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('/Library/Fonts/Arial', size=30)
    for box in boxes:
        draw.rectangle(box, outline='green', width=2)
        draw.text((box[0], box[1]), CLASSES[np.argmax(softmax(resnet_results))], fill='black', font=font)
img.show()
