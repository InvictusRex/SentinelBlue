import os
import cv2
import random
from glob import glob
from tqdm import tqdm
import albumentations as A

TRAIN_IMAGES = "../../Datasets/SentinelBlue/train/images"
TRAIN_LABELS = "../../Datasets/SentinelBlue/train/labels"

BUOY_CLASS_ID = 3
AUG_PER_IMAGE = 1
EPS = 1e-4
IMAGE_EXTS = [".jpg", ".jpeg", ".png"]

augmentations = A.Compose(
    [
        A.RandomScale(scale_limit=(0.5, 1.4), p=1.0),
        A.RandomBrightnessContrast(p=0.7),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.GaussNoise(p=0.3),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.25
    )
)

def clamp(v, lo=EPS, hi=1.0 - EPS):
    return max(lo, min(v, hi))

def sanitize_bbox(b):
    x, y, w, h = b
    w = min(w, 1.0 - EPS)
    h = min(h, 1.0 - EPS)
    x = clamp(x)
    y = clamp(y)
    if w <= EPS or h <= EPS:
        return None
    return [x, y, w, h]

def read_yolo_label(path):
    boxes, classes = [], []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            c = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
            box = sanitize_bbox([x, y, w, h])
            if box is not None:
                boxes.append(box)
                classes.append(c)
    return boxes, classes

def save_yolo_label(path, boxes, classes):
    with open(path, "w") as f:
        for c, b in zip(classes, boxes):
            f.write(f"{int(c)} {' '.join(f'{v:.6f}' for v in b)}\n")

image_files = []
for ext in IMAGE_EXTS:
    image_files.extend(glob(os.path.join(TRAIN_IMAGES, f"*{ext}")))

random.shuffle(image_files)

augmented = 0
skipped = 0

for img_path in tqdm(image_files, desc="Augmenting buoy"):
    base = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(TRAIN_LABELS, base + ".txt")

    if not os.path.exists(lbl_path):
        continue

    boxes, classes = read_yolo_label(lbl_path)

    if BUOY_CLASS_ID not in classes:
        continue

    image = cv2.imread(img_path)
    if image is None:
        continue

    for i in range(AUG_PER_IMAGE):
        try:
            out = augmentations(
                image=image,
                bboxes=boxes,
                class_labels=classes
            )
        except Exception:
            skipped += 1
            continue

        if not out["bboxes"]:
            skipped += 1
            continue

        new_name = f"{base}_aug_buoy_{i}"

        cv2.imwrite(
            os.path.join(TRAIN_IMAGES, new_name + ".jpg"),
            out["image"]
        )

        save_yolo_label(
            os.path.join(TRAIN_LABELS, new_name + ".txt"),
            out["bboxes"],
            out["class_labels"]
        )

        augmented += 1

print(f"Buoy augmentation complete")
print(f"Augmented samples created: {augmented}")
print(f"[i] Samples skipped due to bbox issues: {skipped}")