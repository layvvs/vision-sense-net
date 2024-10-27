import cv2
import cvzone
from ultralytics import YOLO
from pathlib import Path
import torch
import torch
from torchvision import transforms
from PIL import Image
import os

CLASSES = ('angry', 'happy', 'neutral', 'sad', 'surprize')

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

prep = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

MODELS_DIR = Path(os.path.abspath(os.path.abspath(__file__))).parents[1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = YOLO(f'{MODELS_DIR}/models/yolov8n-face.pt').to(device)

resnet50 = torch.load(f'{MODELS_DIR}/models/fine_tuned_resnet_30_10_2024_02_46_26.pt', map_location=torch.device(device))

resnet50 = resnet50.to(device)

resnet50.eval()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()

    for info in model.predict(frame):
        for box in info.boxes:
            rgb_frame = frame[:,:,::-1]
            x1, y1, x2, y2 = list(map(int, box.xyxy[0]))
            cropped_image = Image.fromarray(rgb_frame, mode='RGB').crop((x1, y1, x2, y2))
            proc_img = prep(cropped_image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = resnet50(proc_img)
                _, predicted = torch.max(outputs, 1)
            vibe = CLASSES[predicted.item()]
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, [x1, y1, w, h], l=9, rt=3)
            cv2.putText(frame, vibe, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
