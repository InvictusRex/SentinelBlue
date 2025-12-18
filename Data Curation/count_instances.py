import os
from collections import defaultdict

DATASET_ROOT = "../../Datasets/SeaDronesSee"  

SPLITS = ["train", "val", "test"]

CLASS_NAMES = [
    "boat",
    "buoy",
    "jetski",
    "life_saving_appliances",
    "swimmer"
]

def count_instances():
    counts = defaultdict(int)
    split_counts = {split: defaultdict(int) for split in SPLITS}

    for split in SPLITS:
        labels_dir = os.path.join(DATASET_ROOT, split, "labels")

        if not os.path.isdir(labels_dir):
            print(f"[WARN] Labels directory not found: {labels_dir}")
            continue

        for label_file in os.listdir(labels_dir):
            if not label_file.endswith(".txt"):
                continue

            label_path = os.path.join(labels_dir, label_file)

            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])
                    counts[class_id] += 1
                    split_counts[split][class_id] += 1

    print("TOTAL INSTANCE COUNTS")
    for cid, name in enumerate(CLASS_NAMES):
        print(f"{name:25s}: {counts[cid]}")

    print("\n")
    print("SPLIT-WISE COUNTS")
    for split in SPLITS:
        print(f"[{split.upper()}]")
        for cid, name in enumerate(CLASS_NAMES):
            print(f"{name:25s}: {split_counts[split][cid]}")
        print("\n")

if __name__ == "__main__":
    count_instances()