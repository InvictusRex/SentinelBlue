import shutil
from pathlib import Path

EXTERNAL_ROOT = Path("../../Datasets/Life jacket")

SB_TRAIN_IMAGES = Path("../../Datasets/SentinelBlue/train/images")
SB_TRAIN_LABELS = Path("../../Datasets/SentinelBlue/train/labels")

EXTERNAL_SPLITS = ["train", "val", "test"]

EXTERNAL_CLASS_ID = 0            # 'wear'
SENTINELBLUE_CLASS_ID = 4        # emergency_appliance

def remap_label_and_copy(label_path, image_path):
    new_lines = []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if int(parts[0]) == EXTERNAL_CLASS_ID:
                parts[0] = str(SENTINELBLUE_CLASS_ID)
                new_lines.append(" ".join(parts))

    if not new_lines:
        return False 

    dest_img = SB_TRAIN_IMAGES / image_path.name
    dest_lbl = SB_TRAIN_LABELS / label_path.name

    if not dest_img.exists():
        shutil.copy2(image_path, dest_img)

    if not dest_lbl.exists():
        with open(dest_lbl, "w") as f:
            f.write("\n".join(new_lines))

    return True

total_copied = 0

for split in EXTERNAL_SPLITS:
    img_dir = EXTERNAL_ROOT / split / "images"
    lbl_dir = EXTERNAL_ROOT / split / "labels"

    if not img_dir.exists() or not lbl_dir.exists():
        raise FileNotFoundError(f"Missing images/labels in split: {split}")

    for lbl_path in lbl_dir.glob("*.txt"):
        img_path = img_dir / (lbl_path.stem + ".jpg")
        if not img_path.exists():
            continue

        if remap_label_and_copy(lbl_path, img_path):
            total_copied += 1

print(f"Total life jacket samples copied into SentinelBlue TRAIN: {total_copied}")