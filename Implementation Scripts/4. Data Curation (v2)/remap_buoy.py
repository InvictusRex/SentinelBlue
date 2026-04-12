import argparse
import shutil
from pathlib import Path


SOURCE_CLASS_NAME = "buoy"
TARGET_CLASS_NAME = "emergency_appliance"
TARGET_CLASS_ID = 2

DATA_CURATION_DIR = Path(__file__).resolve().parent
SENTINELBLUE_GITHUB_DIR = DATA_CURATION_DIR.parent
DEFAULT_DATASET_ROOT = SENTINELBLUE_GITHUB_DIR.parent / "Datasets" / "buoy.v5i.yolov11"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remap buoy labels and build a total folder.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to the buoy.v5i.yolov11 dataset folder.",
    )
    return parser.parse_args()


def remap_label_text(label_text: str, label_file: Path) -> str:
    remapped_lines = []

    for raw_line in label_text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        parts = raw_line.split(maxsplit=1)
        try:
            class_id = int(parts[0])
        except ValueError as exc:
            raise ValueError(f"{label_file}: invalid class id in line: {raw_line!r}") from exc

        if class_id != 0:
            raise ValueError(f"{label_file}: expected only class 0 ({SOURCE_CLASS_NAME}), found {class_id}")

        remainder = parts[1] if len(parts) > 1 else ""
        remapped_lines.append(f"{TARGET_CLASS_ID} {remainder}".rstrip())

    return "\n".join(remapped_lines) + ("\n" if remapped_lines else "")


def process_split(dataset_root: Path, split_name: str, total_images_dir: Path, total_labels_dir: Path) -> tuple[int, int]:
    image_dir = dataset_root / split_name / "images"
    label_dir = dataset_root / split_name / "labels"

    if not image_dir.is_dir() or not label_dir.is_dir():
        return 0, 0

    image_paths = sorted(
        [*image_dir.glob("*.jpg"), *image_dir.glob("*.jpeg"), *image_dir.glob("*.png"), *image_dir.glob("*.bmp"), *image_dir.glob("*.webp")]
    )

    copied_images = 0
    copied_labels = 0

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue

        target_stem = f"{split_name}_{image_path.stem}"
        target_image_path = total_images_dir / f"{target_stem}{image_path.suffix.lower()}"
        target_label_path = total_labels_dir / f"{target_stem}.txt"

        shutil.copy2(image_path, target_image_path)
        remapped_text = remap_label_text(label_path.read_text(encoding="utf-8"), label_path)
        target_label_path.write_text(remapped_text, encoding="utf-8")

        copied_images += 1
        copied_labels += 1

    return copied_images, copied_labels


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    total_images_dir = dataset_root / "total" / "images"
    total_labels_dir = dataset_root / "total" / "labels"
    total_images_dir.mkdir(parents=True, exist_ok=True)
    total_labels_dir.mkdir(parents=True, exist_ok=True)

    splits = ("train", "valid", "test")
    overall_images = 0
    overall_labels = 0

    for split_name in splits:
        images_count, labels_count = process_split(dataset_root, split_name, total_images_dir, total_labels_dir)
        overall_images += images_count
        overall_labels += labels_count
        print(f"Finished {split_name} split: copied {images_count} images and {labels_count} labels")

    print(f"Created total folder at {dataset_root / 'total'}")
    print(f"Total copied images: {overall_images}")
    print(f"Total copied labels: {overall_labels}")
    print(f"Remapped class: {SOURCE_CLASS_NAME} -> {TARGET_CLASS_NAME}")


if __name__ == "__main__":
    main()
