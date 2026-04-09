from __future__ import annotations

from collections import Counter
from pathlib import Path


CLASS_NAMES = ["person", "vessel", "emergency_appliance"]
DISPLAY_NAMES = {
    "person": "person",
    "vessel": "vessel",
    "emergency_appliance": "emergencyappliance",
}

DATA_CURATION_DIR = Path(__file__).resolve().parent
SENTINELBLUE_GITHUB_DIR = DATA_CURATION_DIR.parent
DEFAULT_DATASET_ROOT = SENTINELBLUE_GITHUB_DIR.parent / "SentinelBlue"


def count_label_file(label_file: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for raw_line in label_file.read_text(encoding="utf-8").splitlines():
        stripped_line = raw_line.strip()
        if not stripped_line:
            continue

        parts = stripped_line.split(maxsplit=1)
        try:
            class_id = int(parts[0])
        except ValueError as exc:
            raise ValueError(f"{label_file}: invalid class id in line: {raw_line!r}") from exc

        if class_id < 0 or class_id >= len(CLASS_NAMES):
            raise ValueError(f"{label_file}: class id {class_id} is outside the expected range")

        counts[CLASS_NAMES[class_id]] += 1

    return counts


def count_split(dataset_root: Path, split_name: str) -> Counter[str]:
    label_dir = dataset_root / split_name / "labels"
    split_counts: Counter[str] = Counter()

    if not label_dir.is_dir():
        return split_counts

    for label_file in sorted(label_dir.glob("*.txt")):
        split_counts.update(count_label_file(label_file))

    return split_counts


def print_counts(title: str, counts: Counter[str]) -> None:
    print(title)
    for class_name in CLASS_NAMES:
        print(f"{DISPLAY_NAMES[class_name]}: {counts[class_name]}")


def main() -> None:
    dataset_root = DEFAULT_DATASET_ROOT
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    split_names = ["train", "valid", "val", "test"]
    split_counts = {split_name: count_split(dataset_root, split_name) for split_name in split_names}

    total_counts: Counter[str] = Counter()
    for counts in split_counts.values():
        total_counts.update(counts)

    print_counts("Total Instances", total_counts)

    for split_name in split_names:
        if not split_counts[split_name]:
            continue
        print()
        print_counts(f"{split_name} Set Instances", split_counts[split_name])


if __name__ == "__main__":
    main()