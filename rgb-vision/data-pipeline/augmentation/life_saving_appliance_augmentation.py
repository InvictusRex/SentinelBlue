import os
import cv2
import random
import numpy as np
from glob import glob
from tqdm import tqdm
import albumentations as A

TRAIN_IMAGES = "../../Datasets/SentinelBlue/train/images"
TRAIN_LABELS = "../../Datasets/SentinelBlue/train/labels"

EA_CLASS_ID = 4
TARGET_NEW_INSTANCES = 4500
MAX_PASTE_PER_IMAGE = 2
EPS = 1e-4

IMAGE_EXTS = [".jpg", ".jpeg", ".png"]

#Post-paste augmentation
post_aug = A.Compose([
    A.RandomBrightnessContrast(p=0.6),
    A.GaussianBlur(blur_limit=3, p=0.3),
    A.GaussNoise(p=0.3),
])

def clamp(v, lo=EPS, hi=1.0 - EPS):
    return max(lo, min(v, hi))

def read_yolo(path):
    boxes, classes = [], []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            c = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
            boxes.append([x, y, w, h])
            classes.append(c)
    return boxes, classes

def save_yolo(path, boxes, classes):
    with open(path, "w") as f:
        for c, b in zip(classes, boxes):
            f.write(f"{int(c)} {' '.join(f'{v:.6f}' for v in b)}\n")

def yolo_to_xyxy(box, W, H):
    x, y, w, h = box
    x1 = int((x - w / 2) * W)
    y1 = int((y - h / 2) * H)
    x2 = int((x + w / 2) * W)
    y2 = int((y + h / 2) * H)
    return x1, y1, x2, y2

def xyxy_to_yolo(x1, y1, x2, y2, W, H):
    x = ((x1 + x2) / 2) / W
    y = ((y1 + y2) / 2) / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    return [clamp(x), clamp(y), clamp(w), clamp(h)]

#Extracting crops of emergency appliances from training images
ea_crops = []

label_files = glob(os.path.join(TRAIN_LABELS, "*.txt"))
for lbl_path in tqdm(label_files, desc="Collecting appliance crops"):
    base = os.path.splitext(os.path.basename(lbl_path))[0]
    img_path = os.path.join(TRAIN_IMAGES, base + ".jpg")
    if not os.path.exists(img_path):
        continue

    image = cv2.imread(img_path)
    if image is None:
        continue

    H, W = image.shape[:2]
    boxes, classes = read_yolo(lbl_path)

    for box, cls in zip(boxes, classes):
        if cls == EA_CLASS_ID:
            x1, y1, x2, y2 = yolo_to_xyxy(box, W, H)
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                ea_crops.append(crop)

print(f"[INFO] Collected {len(ea_crops)} emergency appliance crops")

#Copy Paste part
image_files = glob(os.path.join(TRAIN_IMAGES, "*.jpg"))
random.shuffle(image_files)

created = 0

for img_path in tqdm(image_files, desc="Copy-paste augmentation"):
    if created >= TARGET_NEW_INSTANCES:
        break

    base = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(TRAIN_LABELS, base + ".txt")

    if not os.path.exists(lbl_path):
        continue

    image = cv2.imread(img_path)
    if image is None:
        continue

    H, W = image.shape[:2]
    boxes, classes = read_yolo(lbl_path)

    paste_count = random.randint(1, MAX_PASTE_PER_IMAGE)

    for _ in range(paste_count):
        if created >= TARGET_NEW_INSTANCES:
            break

        crop = random.choice(ea_crops)
        ch, cw = crop.shape[:2]

        if ch >= H or cw >= W:
            continue

        x1 = random.randint(0, W - cw)
        y1 = random.randint(0, H - ch)

        image[y1:y1+ch, x1:x1+cw] = crop

        new_box = xyxy_to_yolo(x1, y1, x1+cw, y1+ch, W, H)
        boxes.append(new_box)
        classes.append(EA_CLASS_ID)

        created += 1

    augmented = post_aug(image=image)["image"]

    new_name = f"{base}_aug_ea_{random.randint(1000,9999)}"
    cv2.imwrite(os.path.join(TRAIN_IMAGES, new_name + ".jpg"), augmented)
    save_yolo(os.path.join(TRAIN_LABELS, new_name + ".txt"), boxes, classes)

print(f"Emergency appliance instances added: {created}")