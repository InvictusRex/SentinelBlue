import shutil
from pathlib import Path

TRAIN_IMAGES = Path("../../Datasets/SentinelBlue/train/images")
TRAIN_LABELS = Path("../../Datasets/SentinelBlue/train/labels")

VAL_IMAGES = Path("../../Datasets/SentinelBlue/val/images")
VAL_LABELS = Path("../../Datasets/SentinelBlue/val/labels")

TEST_IMAGES = Path("../../Datasets/SentinelBlue/test/images")
TEST_LABELS = Path("../../Datasets/SentinelBlue/test/labels")

EMERGENCY_CLASS_ID = 4

def label_contains_emergency(label_path):
    with open(label_path, "r") as f:
        for line in f:
            if line.strip() and int(line.split()[0]) == EMERGENCY_CLASS_ID:
                return True
    return False

def copy_emergency_samples(images_dir, labels_dir):
    copied = 0

    for lbl_path in labels_dir.glob("*.txt"):
        if not label_contains_emergency(lbl_path):
            continue

        img_path = images_dir / (lbl_path.stem + ".jpg")
        if not img_path.exists():
            continue

        dest_img = TRAIN_IMAGES / img_path.name
        dest_lbl = TRAIN_LABELS / lbl_path.name

        if not dest_img.exists():
            shutil.copy2(img_path, dest_img)
        if not dest_lbl.exists():
            shutil.copy2(lbl_path, dest_lbl)

        copied += 1

    return copied

val_copied = copy_emergency_samples(VAL_IMAGES, VAL_LABELS)
test_copied = copy_emergency_samples(TEST_IMAGES, TEST_LABELS)

print(f"Copied {val_copied} emergency_appliance samples from VAL to TRAIN")
print(f"Copied {test_copied} emergency_appliance samples from TEST to TRAIN")
print(f"Total emergency_appliance samples copied: {val_copied + test_copied}")