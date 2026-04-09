import argparse
from collections import Counter
from pathlib import Path


SOURCE_CLASS_NAMES = ["boat", "buoy", "jetski", "life_saving_appliances", "swimmer"]
NEW_CLASS_NAMES = ["person", "vessel", "emergency_appliance"]

SOURCE_TO_TARGET_NAME = {
    "boat": "vessel",
    "jetski": "vessel",
    "buoy": "emergency_appliance",
    "life_saving_appliances": "emergency_appliance",
    "swimmer": "person",
}

TARGET_NAME_TO_ID = {name: index for index, name in enumerate(NEW_CLASS_NAMES)}

DATA_CURATION_DIR = Path(__file__).resolve().parent
SENTINELBLUE_GITHUB_DIR = DATA_CURATION_DIR.parent
DEFAULT_DATASET_ROOT = SENTINELBLUE_GITHUB_DIR.parent / "SentinelBlue"


def default_dataset_root() -> Path:
    return DEFAULT_DATASET_ROOT


def remap_label_file(label_file: Path) -> Counter[str]:
    counts = Counter()
    remapped_lines = []

    for raw_line in label_file.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        parts = raw_line.split(maxsplit=1)
        if len(parts) == 0:
            continue

        old_class_id = int(parts[0])
        if old_class_id < 0 or old_class_id >= len(SOURCE_CLASS_NAMES):
            raise ValueError(f"{label_file}: class id {old_class_id} is outside the expected SeaDronesSee range")

        source_name = SOURCE_CLASS_NAMES[old_class_id]
        target_name = SOURCE_TO_TARGET_NAME[source_name]
        remainder = parts[1] if len(parts) > 1 else ""

        remapped_lines.append(f"{TARGET_NAME_TO_ID[target_name]} {remainder}".rstrip())
        counts[target_name] += 1

    label_file.write_text("\n".join(remapped_lines) + ("\n" if remapped_lines else ""), encoding="utf-8")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Remap SeaDronesSee YOLO labels to 3 classes.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_dataset_root(),
        help="Path to the SentinelBlue dataset root directory.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    splits = ("train", "valid", "val", "test")
    label_dirs = []
    for split_name in splits:
        label_dir = dataset_root / split_name / "labels"
        if label_dir.is_dir():
            label_dirs.append((split_name, label_dir))

    if not label_dirs:
        raise FileNotFoundError(
            f"No label directories found under {dataset_root}. Expected train/labels, valid/labels, val/labels, or test/labels."
        )

    total_counts: Counter[str] = Counter()
    total_files = 0
    total_changed = 0

    for split_name, label_dir in label_dirs:
        split_files = sorted(label_dir.glob("*.txt"))
        if not split_files:
            continue

        print(f"Beginning {split_name} split")

        split_counts = Counter()
        split_changed = 0

        for label_file in split_files:
            file_counts = remap_label_file(label_file)
            split_counts.update(file_counts)
            split_changed += sum(file_counts.values())

        total_counts.update(split_counts)
        total_files += len(split_files)
        total_changed += split_changed
        print(f"Finished {split_name} split: changed {split_changed} annotations across {len(split_files)} label files")

    if total_files == 0:
        raise FileNotFoundError(f"No .txt label files found under {dataset_root}")

    print(f"Processed {total_files} label files under {dataset_root}")
    print(f"Total changed annotations: {total_changed}")
    for class_name in NEW_CLASS_NAMES:
        print(f"{class_name}: {total_counts[class_name]}")


if __name__ == "__main__":
    main()