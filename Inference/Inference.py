import cv2
from ultralytics import YOLO
import torch

MODEL_PATH = r'../Models Weights/Small Models/YOLO-28s.pt'                
VIDEO_PATH = r'../Inference/Test Videos/swimmers.mp4'        
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Using device: {DEVICE}')

model = YOLO(MODEL_PATH).to(DEVICE)

model.model.names = {
    0: "boat",                 
    1: "buoy",                   
    2: "jetski",                 
    3: "emergency_appliance",    
    4: "person"
}

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, device=DEVICE)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()