from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = ["person", "vessel", "emergency_appliance"]
CLASS_COLORS = {
    "person": (255, 128, 0),
    "vessel": (0, 200, 255),
    "emergency_appliance": (0, 255, 128),
}

DATA_CURATION_DIR = Path(__file__).resolve().parent
SENTINELBLUE_GITHUB_DIR = DATA_CURATION_DIR.parent
DEFAULT_DATASET_ROOT = SENTINELBLUE_GITHUB_DIR.parent / "SentinelBlue"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View train-set bounding boxes with OpenCV.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to the SentinelBlue dataset root directory.",
    )
    return parser.parse_args()


def load_image_paths(dataset_root: Path) -> list[Path]:
    image_dir = dataset_root / "train" / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Train image directory not found: {image_dir}")

    image_paths = sorted(
        [*image_dir.glob("*.jpg"), *image_dir.glob("*.jpeg"), *image_dir.glob("*.png")]
    )
    if not image_paths:
        raise FileNotFoundError(f"No train images found in: {image_dir}")

    return image_paths


def label_path_for_image(image_path: Path, dataset_root: Path) -> Path:
    return dataset_root / "train" / "labels" / f"{image_path.stem}.txt"


def read_boxes(label_path: Path, image_width: int, image_height: int) -> list[tuple[str, int, int, int, int]]:
    boxes: list[tuple[str, int, int, int, int]] = []
    if not label_path.is_file():
        return boxes

    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        stripped_line = raw_line.strip()
        if not stripped_line:
            continue

        parts = stripped_line.split()
        if len(parts) < 5:
            continue

        class_id = int(parts[0])
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            continue

        x_center = float(parts[1]) * image_width
        y_center = float(parts[2]) * image_height
        box_width = float(parts[3]) * image_width
        box_height = float(parts[4]) * image_height

        x1 = int(round(x_center - box_width / 2))
        y1 = int(round(y_center - box_height / 2))
        x2 = int(round(x_center + box_width / 2))
        y2 = int(round(y_center + box_height / 2))

        boxes.append((CLASS_NAMES[class_id], x1, y1, x2, y2))

    return boxes


def draw_boxes(image, boxes):
    for class_name, x1, y1, x2, y2 in boxes:
        color = CLASS_COLORS[class_name]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

        label_text = class_name
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        text_x = max(0, x1)
        text_y = max(text_height + 10, y1)
        box_top = max(0, text_y - text_height - baseline - 8)
        box_left = max(0, text_x)
        box_right = min(image.shape[1] - 1, text_x + text_width + 12)
        box_bottom = min(image.shape[0] - 1, text_y + 6)

        cv2.rectangle(image, (box_left, box_top), (box_right, box_bottom), color, -1)
        cv2.putText(
            image,
            label_text,
            (text_x + 6, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )


def fit_to_canvas(image, canvas_width: int, canvas_height: int):
    image_height, image_width = image.shape[:2]
    scale = min(canvas_width / image_width, canvas_height / image_height, 1.0)
    resized_width = max(1, int(round(image_width * scale)))
    resized_height = max(1, int(round(image_height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=resized.dtype)
    offset_x = (canvas_width - resized_width) // 2
    offset_y = (canvas_height - resized_height) // 2
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    return canvas


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    image_paths = load_image_paths(dataset_root)
    window_name = "SentinelBlue Train Boxes"
    canvas_width = 2560
    canvas_height = 1600

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, canvas_width, canvas_height)

    index = 0
    while True:
        image_path = image_paths[index]
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        image_height, image_width = image.shape[:2]
        boxes = read_boxes(label_path_for_image(image_path, dataset_root), image_width, image_height)
        draw_boxes(image, boxes)

        display = fit_to_canvas(image, canvas_width, canvas_height)
        header = f"{index + 1}/{len(image_paths)}  {image_path.name}  |  a: previous  d: next  q: quit"
        cv2.putText(display, header, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        if key == ord("a"):
            index = (index - 1) % len(image_paths)
            continue
        if key == ord("d"):
            index = (index + 1) % len(image_paths)
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()