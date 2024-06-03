import cv2
import cvzone
from ultralytics import YOLO
from pathlib import Path
import os

MODELS_DIR = Path(os.path.abspath(os.path.abspath(__file__))).parents[1]

model = YOLO(f'{MODELS_DIR}/models/yolov8n-face.pt')

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()

    for info in model.predict(frame):
        for box in info.boxes:
            x1, y1, x2, y2 = list(map(int, box.xyxy[0]))
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, [x1, y1, w, h], l=9, rt=3)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
