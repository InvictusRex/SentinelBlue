import shutil
from pathlib import Path

EXTERNAL_ROOT = Path("../../Datasets/Life Saving Tools")

SB_TRAIN_IMAGES = Path("../../Datasets/SentinelBlue/train/images")
SB_TRAIN_LABELS = Path("../../Datasets/SentinelBlue/train/labels")

EXTERNAL_SPLITS = ["train", "val", "test"]

VALID_EXTERNAL_CLASSES = {0, 1, 2}   # LPU, life-raft, life-ring
DROP_EXTERNAL_CLASS = 3              # orion

SENTINELBLUE_CLASS_ID = 4             # emergency_appliance

def filter_and_remap_label(label_path):

    new_lines = []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            class_id = int(parts[0])

            if class_id in VALID_EXTERNAL_CLASSES:
                parts[0] = str(SENTINELBLUE_CLASS_ID)
                new_lines.append(" ".join(parts))

    return new_lines

total_copied = 0
total_skipped_orion_only = 0

for split in EXTERNAL_SPLITS:
    img_dir = EXTERNAL_ROOT / split / "images"
    lbl_dir = EXTERNAL_ROOT / split / "labels"

    if not img_dir.exists() or not lbl_dir.exists():
        raise FileNotFoundError(f"Missing images/labels in split: {split}")

    for lbl_path in lbl_dir.glob("*.txt"):
        remapped_lines = filter_and_remap_label(lbl_path)

        if not remapped_lines:
            total_skipped_orion_only += 1
            continue

        img_path = img_dir / (lbl_path.stem + ".jpg")
        if not img_path.exists():
            continue

        dest_img = SB_TRAIN_IMAGES / img_path.name
        dest_lbl = SB_TRAIN_LABELS / lbl_path.name

        if not dest_img.exists():
            shutil.copy2(img_path, dest_img)

        if not dest_lbl.exists():
            with open(dest_lbl, "w") as f:
                f.write("\n".join(remapped_lines))

        total_copied += 1

print(f"Life-saving appliance samples copied into TRAIN: {total_copied}")
print(f"[i] Images skipped (only 'orion'): {total_skipped_orion_only}")