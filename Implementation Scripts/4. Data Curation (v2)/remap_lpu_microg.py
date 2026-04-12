import argparse
import shutil
from pathlib import Path


TARGET_CLASS_ID = 2
SOURCE_CLASSES = {0, 1, 2}

DATA_CURATION_DIR = Path(__file__).resolve().parent
SENTINELBLUE_GITHUB_DIR = DATA_CURATION_DIR.parent
DEFAULT_DATASET_ROOT = SENTINELBLUE_GITHUB_DIR.parent / "Datasets" / "LPU-microg.v2i.yolov11"


def parse_args():
    parser = argparse.ArgumentParser(description="Remap LPU-Microg labels into SentinelBlue emergency_appliance labels.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to the LPU-microg.v2i.yolov11 dataset folder.",
    )
    return parser.parse_args()


def remap_label_file(label_file):
    kept_lines = []
    kept_boxes = 0

    for line in label_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split(maxsplit=1)
        class_id = int(parts[0])

        if class_id not in SOURCE_CLASSES:
            continue

        rest = parts[1] if len(parts) > 1 else ""
        kept_lines.append(f"{TARGET_CLASS_ID} {rest}".rstrip())
        kept_boxes += 1

    return kept_lines, kept_boxes


def process_split(dataset_root, split_name, total_images_dir, total_labels_dir):
    image_dir = dataset_root / split_name / "images"
    label_dir = dataset_root / split_name / "labels"

    if not image_dir.is_dir() or not label_dir.is_dir():
        return 0, 0, 0

    image_files = sorted(
        list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.jpeg"))
        + list(image_dir.glob("*.png"))
        + list(image_dir.glob("*.bmp"))
        + list(image_dir.glob("*.webp"))
    )

    copied_images = 0
    copied_labels = 0
    kept_boxes = 0

    for image_file in image_files:
        label_file = label_dir / f"{image_file.stem}.txt"
        if not label_file.is_file():
            continue

        lines, box_count = remap_label_file(label_file)
        if box_count == 0:
            continue

        new_stem = f"{split_name}_{image_file.stem}"
        new_image_file = total_images_dir / f"{new_stem}{image_file.suffix.lower()}"
        new_label_file = total_labels_dir / f"{new_stem}.txt"

        shutil.copy2(image_file, new_image_file)
        new_label_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        copied_images += 1
        copied_labels += 1
        kept_boxes += box_count

    return copied_images, copied_labels, kept_boxes


def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    total_images_dir = dataset_root / "total" / "images"
    total_labels_dir = dataset_root / "total" / "labels"
    total_images_dir.mkdir(parents=True, exist_ok=True)
    total_labels_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_labels = 0
    total_boxes = 0

    for split_name in ("train", "valid", "test"):
        print(f"Starting {split_name} split")
        images, labels, boxes = process_split(dataset_root, split_name, total_images_dir, total_labels_dir)
        total_images += images
        total_labels += labels
        total_boxes += boxes
        print(f"Finished {split_name} split: copied {images} images, {labels} labels, kept {boxes} boxes")

    print(f"Created total folder at {dataset_root / 'total'}")
    print(f"Total copied images: {total_images}")
    print(f"Total copied labels: {total_labels}")
    print(f"Total kept boxes: {total_boxes}")


if __name__ == "__main__":
    main()
