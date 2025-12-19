import os
from collections import defaultdict

DATASET_ROOT = "../../Datasets/sentinelblue"
SPLITS = ["train", "val", "test"]

CLASS_NAMES = [
    "person",
    "boat",
    "jetski",
    "buoy",
    "emergency_appliance"
]

def count_instances():
    counts = defaultdict(int)
    split_counts = {split: defaultdict(int) for split in SPLITS}
    total_instances = 0

    for split in SPLITS:
        labels_dir = os.path.join(DATASET_ROOT, split, "labels")

        if not os.path.isdir(labels_dir):
            print(f"[WARN] Missing labels dir: {labels_dir}")
            continue

        for label_file in os.listdir(labels_dir):
            if not label_file.endswith(".txt"):
                continue

            with open(os.path.join(labels_dir, label_file), "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])
                    if class_id < 0 or class_id >= len(CLASS_NAMES):
                        continue

                    counts[class_id] += 1
                    split_counts[split][class_id] += 1
                    total_instances += 1

    print("\nTOTAL INSTANCE COUNTS")
    for cid, name in enumerate(CLASS_NAMES):
        print(f"{name:25s}: {counts[cid]}")

    print("\nSPLIT-WISE COUNTS")
    for split in SPLITS:
        print(f"[{split.upper()}]")
        for cid, name in enumerate(CLASS_NAMES):
            print(f"{name:25s}: {split_counts[split][cid]}")
        print()

    print(f"TOTAL OBJECT INSTANCES: {total_instances}")

if __name__ == "__main__":
    count_instances()