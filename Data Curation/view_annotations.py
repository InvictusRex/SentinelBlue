import os
import cv2

DATASET_ROOT = "../../Datasets/sentinelblue"
SPLIT = "train"   # change to val / test if needed

CLASS_NAMES = [
    "person",
    "boat",
    "jetski",
    "buoy",
    "emergency_appliance"
]

IMG_DIR = os.path.join(DATASET_ROOT, SPLIT, "images")
LBL_DIR = os.path.join(DATASET_ROOT, SPLIT, "labels")

IMAGE_EXTS = (".jpg", ".png", ".jpeg")

MAX_WIDTH = 1280
MAX_HEIGHT = 720

images = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(IMAGE_EXTS)])
idx = 0

def resize_for_display(img):
    h, w = img.shape[:2]
    scale = min(MAX_WIDTH / w, MAX_HEIGHT / h, 1.0)
    return cv2.resize(img, (int(w * scale), int(h * scale)))

def draw_boxes(img, label_path):
    h, w = img.shape[:2]

    if not os.path.exists(label_path):
        return img

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id, x, y, bw, bh = map(float, parts)
            class_id = int(class_id)

            if class_id < 0 or class_id >= len(CLASS_NAMES):
                continue

            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                CLASS_NAMES[class_id],
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

    return img

cv2.namedWindow("SentinelBlue Viewer", cv2.WINDOW_NORMAL)

while True:
    img_path = os.path.join(IMG_DIR, images[idx])
    lbl_path = os.path.join(LBL_DIR, os.path.splitext(images[idx])[0] + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        idx = (idx + 1) % len(images)
        continue

    vis = draw_boxes(img.copy(), lbl_path)
    vis = resize_for_display(vis)

    cv2.imshow("SentinelBlue Viewer", vis)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('n'):
        idx = (idx + 1) % len(images)
    elif key == ord('p'):
        idx = (idx - 1) % len(images)
    elif key == ord('q'):
        break

cv2.destroyAllWindows()