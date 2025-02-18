from ultralytics import YOLO
import torch
import onnx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = YOLO('/Users/layvvs/Desktop/Studying/BMSTU/Diploma-2024-2025/code/vision-sense-net/models/yolov8n-face.pt')

model.eval()

model.export(format='onnx')

onnx_model = onnx.load("/Users/layvvs/Desktop/Studying/BMSTU/Diploma-2024-2025/code/vision-sense-net/models/yolov8n-face.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")
