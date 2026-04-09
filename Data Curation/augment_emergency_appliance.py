import argparse
import random
from pathlib import Path

import cv2
import numpy as np


TARGET_CLASS_ID = 2
TARGET_COUNT = 10000

DATA_CURATION_DIR = Path(__file__).resolve().parent
SENTINELBLUE_GITHUB_DIR = DATA_CURATION_DIR.parent
DEFAULT_DATASET_ROOT = SENTINELBLUE_GITHUB_DIR.parent / "SentinelBlue"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATASET_ROOT / "train_augmented"


def parse_args():
    parser = argparse.ArgumentParser(description="Augment emergency_appliance instances up to a target count.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-count", type=int, default=TARGET_COUNT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def find_image_path(image_dir, stem):
    for extension in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        candidate = image_dir / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No image found for {stem} in {image_dir}")


def read_label_file(label_path):
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        boxes.append((class_id, x_center, y_center, width, height))
    return boxes


def write_label_file(label_path, boxes):
    lines = []
    for class_id, x_center, y_center, width, height in boxes:
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def count_target_instances(label_paths):
    total = 0
    for label_path in label_paths:
        for class_id, *_ in read_label_file(label_path):
            if class_id == TARGET_CLASS_ID:
                total += 1
    return total


def yolo_to_pixels(box, image_width, image_height):
    _, x_center, y_center, width, height = box
    x_center *= image_width
    y_center *= image_height
    width *= image_width
    height *= image_height
    x1 = x_center - width / 2.0
    y1 = y_center - height / 2.0
    x2 = x_center + width / 2.0
    y2 = y_center + height / 2.0
    return x1, y1, x2, y2


def pixels_to_yolo(class_id, x1, y1, x2, y2, image_width, image_height):
    x1 = max(0.0, min(float(image_width - 1), x1))
    y1 = max(0.0, min(float(image_height - 1), y1))
    x2 = max(0.0, min(float(image_width - 1), x2))
    y2 = max(0.0, min(float(image_height - 1), y2))

    if x2 <= x1 or y2 <= y1:
        return None

    width = x2 - x1
    height = y2 - y1
    x_center = x1 + width / 2.0
    y_center = y1 + height / 2.0

    return (
        class_id,
        x_center / image_width,
        y_center / image_height,
        width / image_width,
        height / image_height,
    )


def transform_boxes(boxes, matrix, image_width, image_height):
    updated = []
    for class_id, x_center, y_center, width, height in boxes:
        x1, y1, x2, y2 = yolo_to_pixels((class_id, x_center, y_center, width, height), image_width, image_height)
        points = np.array(
            [[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]],
            dtype=np.float32,
        )
        transformed = (matrix @ points.T).T
        xs = transformed[:, 0]
        ys = transformed[:, 1]
        box = pixels_to_yolo(class_id, xs.min(), ys.min(), xs.max(), ys.max(), image_width, image_height)
        if box is not None:
            updated.append(box)
    return updated


def flip_horizontal(image, boxes):
    flipped = cv2.flip(image, 1)
    image_height, image_width = image.shape[:2]
    updated_boxes = []
    for class_id, x_center, y_center, width, height in boxes:
        updated_boxes.append((class_id, 1.0 - x_center, y_center, width, height))
    return flipped, updated_boxes


def affine_transform(image, boxes):
    image_height, image_width = image.shape[:2]
    angle = random.uniform(-10.0, 10.0)
    scale = random.uniform(0.92, 1.08)
    shift_x = random.uniform(-0.06, 0.06) * image_width
    shift_y = random.uniform(-0.06, 0.06) * image_height

    matrix = cv2.getRotationMatrix2D((image_width / 2.0, image_height / 2.0), angle, scale)
    matrix[0, 2] += shift_x
    matrix[1, 2] += shift_y

    transformed_image = cv2.warpAffine(
        image,
        matrix,
        (image_width, image_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    full_matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
    updated_boxes = transform_boxes(boxes, full_matrix, image_width, image_height)
    return transformed_image, updated_boxes


def adjust_brightness_contrast(image):
    alpha = random.uniform(0.85, 1.15)
    beta = random.randint(-20, 20)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def adjust_hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-6, 6)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.90, 1.10), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.85, 1.15), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def add_gaussian_blur(image):
    kernel = random.choice([3, 5])
    return cv2.GaussianBlur(image, (kernel, kernel), 0)


def add_motion_blur(image):
    size = random.choice([5, 7, 9])
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    kernel /= kernel.sum()
    angle = random.uniform(0.0, 180.0)
    matrix = cv2.getRotationMatrix2D((size / 2.0 - 0.5, size / 2.0 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (size, size))
    kernel = kernel / max(kernel.sum(), 1e-6)
    return cv2.filter2D(image, -1, kernel)


def add_noise(image):
    noise = np.random.normal(0, random.uniform(6.0, 14.0), image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_appearance_augments(image):
    choices = [adjust_brightness_contrast, adjust_hsv, add_gaussian_blur, add_motion_blur, add_noise]
    random.shuffle(choices)
    image = choices[0](image)
    if random.random() < 0.5:
        image = choices[1](image)
    return image


def augment_sample(image, boxes):
    if random.random() < 0.5:
        image, boxes = flip_horizontal(image, boxes)

    if random.random() < 0.8:
        image, boxes = affine_transform(image, boxes)

    image = apply_appearance_augments(image)
    return image, boxes


def load_source_samples(dataset_root):
    image_dir = dataset_root / "train" / "images"
    label_dir = dataset_root / "train" / "labels"

    samples = []
    label_paths = []

    for label_path in sorted(label_dir.glob("*.txt")):
        boxes = read_label_file(label_path)
        if not any(class_id == TARGET_CLASS_ID for class_id, *_ in boxes):
            continue

        image_path = find_image_path(image_dir, label_path.stem)
        samples.append((image_path, label_path, boxes))
        label_paths.append(label_path)

    return samples, label_paths


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    samples, label_paths = load_source_samples(dataset_root)
    if not samples:
        raise FileNotFoundError("No train images with emergency_appliance labels were found.")

    current_count = count_target_instances(label_paths)
    if current_count >= args.target_count:
        print(f"Current emergency_appliance count is already {current_count}.")
        return

    output_images_dir = output_root / "images"
    output_labels_dir = output_root / "labels"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"Current emergency_appliance count: {current_count}")
    print(f"Target emergency_appliance count: {args.target_count}")
    print(f"Saving augmented files to: {output_root}")

    created = 0
    attempts = 0
    max_attempts = (args.target_count - current_count) * 20

    while current_count < args.target_count and attempts < max_attempts:
        attempts += 1
        image_path, label_path, boxes = random.choice(samples)
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        augmented_image, augmented_boxes = augment_sample(image, boxes)
        if not augmented_boxes:
            continue

        next_index = created + 1
        stem = f"aug_{next_index:05d}_{image_path.stem}"
        out_image_path = output_images_dir / f"{stem}{image_path.suffix.lower()}"
        out_label_path = output_labels_dir / f"{stem}.txt"

        cv2.imwrite(str(out_image_path), augmented_image)
        write_label_file(out_label_path, augmented_boxes)

        current_count += sum(1 for class_id, *_ in augmented_boxes if class_id == TARGET_CLASS_ID)
        created += 1

        if created % 50 == 0:
            print(f"Created {created} augmented images so far. Current count: {current_count}")

    print(f"Finished after {created} augmented images.")
    print(f"Final emergency_appliance count: {current_count}")
    if current_count < args.target_count:
        print(f"Stopped early after {attempts} attempts because the target was not reached.")


if __name__ == "__main__":
    main()
