import cv2
from ultralytics import YOLO
import torch

MODEL_PATH = r'../Models Weights/Small Models/YOLO-28s.pt'                
VIDEO_PATH = r'../Inference/Test Videos/multi.mp4'        
OUTPUT_PATH = r'../Inference/Output Videos/multi.mp4'

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

# 🔹 Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 🔹 Initialize VideoWriter (MP4 codec)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

# Display window
cv2.namedWindow("YOLO Inference", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Inference", 960, 540)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device=DEVICE)
    annotated_frame = results[0].plot()

    # 🔹 Write ORIGINAL RESOLUTION annotated frame to video
    out.write(annotated_frame)

    # 🔹 Resize only for display (not for saving)
    display_frame = cv2.resize(annotated_frame, (960, 540))
    cv2.imshow("YOLO Inference", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()   # 🔥 IMPORTANT: finalize file
cv2.destroyAllWindows()

print(f"Saved annotated video to: {OUTPUT_PATH}")