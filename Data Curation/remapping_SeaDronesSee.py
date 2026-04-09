import os

DATASET = "../../SentinelBlue"
SPLITS = ["train", "val", "test"]

CLASS_ID_MAP = {
    0: 1,  # boat -> vessel
    1: 2,  # buoy -> emergency_appliance
    2: 1,  # jetski -> vessel
    3: 2,  # life_saving_appliances -> emergency_appliance
    4: 0,  # swimmer -> person
}

def remap_labels_in_place(label_path):
    with open(label_path, "r") as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        old_id = int(parts[0])
        if old_id not in CLASS_ID_MAP:
            continue

        new_id = CLASS_ID_MAP[old_id]
        parts[0] = str(new_id)
        new_lines.append(" ".join(parts))

    with open(label_path, "w") as f:
        f.write("\n".join(new_lines))

def process_split(split):
    label_dir = os.path.join(DATASET, split, "labels")

    if not os.path.isdir(label_dir):
        print(f"Skipping {split}: labels directory not found at {label_dir}")
        return

    for fname in os.listdir(label_dir):
        if not fname.lower().endswith(".txt"):
            continue

        label_path = os.path.join(label_dir, fname)
        remap_labels_in_place(label_path)

def main():
    for split in SPLITS:
        print(f"Processing {split}...")
        process_split(split)

    print("Class remapping complete (in place).")

if __name__ == "__main__":
    main()