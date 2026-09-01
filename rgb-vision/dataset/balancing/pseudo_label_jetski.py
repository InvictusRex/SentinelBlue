import os
from ultralytics import YOLO
import cv2

MODEL_PATH = "../best.pt"   
DATASET_ROOT = "../../Datasets/jet ski detection/train"

IMG_DIR = os.path.join(DATASET_ROOT, "images")
LBL_DIR = os.path.join(DATASET_ROOT, "labels")

CONF_THRESHOLD = 0.65

OLD_JETSKI_CLASS_ID = 3

NEW_JETSKI_CLASS_ID = 2

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

model = YOLO(MODEL_PATH)

def is_label_empty(label_path):
    if not os.path.exists(label_path):
        return True
    with open(label_path, "r") as f:
        return len(f.readlines()) == 0

def write_yolo_label(label_path, boxes, img_w, img_h):
    lines = []
    for (x1, y1, x2, y2) in boxes:
        xc = ((x1 + x2) / 2) / img_w
        yc = ((y1 + y2) / 2) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h

        lines.append(
            f"{NEW_JETSKI_CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
        )

    with open(label_path, "w") as f:
        f.write("\n".join(lines))

def main():
    updated = 0
    skipped = 0

    for img_name in os.listdir(IMG_DIR):
        if not img_name.lower().endswith(IMAGE_EXTS):
            continue

        base = os.path.splitext(img_name)[0]
        img_path = os.path.join(IMG_DIR, img_name)
        lbl_path = os.path.join(LBL_DIR, base + ".txt")

        if not is_label_empty(lbl_path):
            skipped += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        results = model(img, conf=CONF_THRESHOLD, verbose=False)

        boxes = []
        for r in results:
            if r.boxes is None:
                continue

            for b in r.boxes:
                cls = int(b.cls.item())
                conf = float(b.conf.item())

                if cls == OLD_JETSKI_CLASS_ID and conf >= CONF_THRESHOLD:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    boxes.append((x1, y1, x2, y2))

        if boxes:
            write_yolo_label(lbl_path, boxes, w, h)
            updated += 1

    print("Pseudo-labeling complete.")
    print(f"Updated label files : {updated}")
    print(f"Skipped (already labeled): {skipped}")

if __name__ == "__main__":
    main()