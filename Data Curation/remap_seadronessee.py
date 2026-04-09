from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable


SOURCE_CLASS_NAMES = [
    "boat",
    "buoy",
    "jetski",
    "life_saving_appliances",
    "swimmer",
]

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


def candidate_label_dirs(dataset_root: Path) -> list[tuple[str, Path]]:
    label_dirs: list[tuple[str, Path]] = []
    for split_name in ("train", "valid", "val", "test"):
        label_dir = dataset_root / split_name / "labels"
        if label_dir.is_dir():
            label_dirs.append((split_name, label_dir))
    return label_dirs


def remap_label_file(label_file: Path) -> Counter[str]:
    updated_lines: list[str] = []
    class_counts: Counter[str] = Counter()

    for raw_line in label_file.read_text(encoding="utf-8").splitlines():
        stripped_line = raw_line.strip()
        if not stripped_line:
            updated_lines.append("")
            continue

        parts = stripped_line.split(maxsplit=1)
        if not parts:
            updated_lines.append("")
            continue

        try:
            old_class_id = int(parts[0])
        except ValueError as exc:
            raise ValueError(f"{label_file}: invalid class id in line: {raw_line!r}") from exc

        if old_class_id < 0 or old_class_id >= len(SOURCE_CLASS_NAMES):
            raise ValueError(
                f"{label_file}: class id {old_class_id} is outside the expected SeaDronesSee range"
            )

        source_name = SOURCE_CLASS_NAMES[old_class_id]
        target_name = SOURCE_TO_TARGET_NAME[source_name]
        target_class_id = TARGET_NAME_TO_ID[target_name]

        remainder = parts[1] if len(parts) > 1 else ""
        updated_lines.append(f"{target_class_id} {remainder}".rstrip())
        class_counts[target_name] += 1

    label_file.write_text("\n".join(updated_lines) + ("\n" if updated_lines else ""), encoding="utf-8")
    return class_counts


def iter_label_files(label_dirs: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for label_dir in label_dirs:
        files.extend(sorted(label_dir.glob("*.txt")))
    return files


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

    label_dirs = candidate_label_dirs(dataset_root)
    if not label_dirs:
        raise FileNotFoundError(
            f"No label directories found under {dataset_root}. Expected train/labels, valid/labels, val/labels, or test/labels."
        )

    total_counts: Counter[str] = Counter()
    total_files = 0
    total_changed = 0

    for split_name, label_dir in label_dirs:
        split_files = iter_label_files([label_dir])
        if not split_files:
            continue

        print(f"Beginning {split_name} split")

        split_counts: Counter[str] = Counter()
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