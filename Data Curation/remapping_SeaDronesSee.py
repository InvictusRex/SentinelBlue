import os
import shutil

SRC_DATASET = "../../Datasets/SeaDronesSee"              # SeaDronesSee root
DST_DATASET = "../../Datasets/SentinelBlue" # New curated dataset

SPLITS = ["train", "val", "test"]

# Old SeaDronesSee class IDs -> New SentinelBlue class IDs
# Old classes (from Roboflow):
# 0: boat
# 1: buoy
# 2: jetski
# 3: life_saving_appliances
# 4: swimmer

CLASS_ID_MAP = {
    0: 1,  # boat -> boat
    1: 3,  # buoy -> buoy
    2: 2,  # jetski -> jetski
    3: 4,  # life_saving_appliances -> emergency_appliance
    4: 0   # swimmer -> person
}

def ensure_dirs():
    for split in SPLITS:
        os.makedirs(os.path.join(DST_DATASET, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(DST_DATASET, split, "labels"), exist_ok=True)

def remap_labels(src_label_path, dst_label_path):
    with open(src_label_path, "r") as f:
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

    with open(dst_label_path, "w") as f:
        f.write("\n".join(new_lines))

def process_split(split):
    src_img_dir = os.path.join(SRC_DATASET, split, "images")
    src_lbl_dir = os.path.join(SRC_DATASET, split, "labels")

    dst_img_dir = os.path.join(DST_DATASET, split, "images")
    dst_lbl_dir = os.path.join(DST_DATASET, split, "labels")

    for fname in os.listdir(src_img_dir):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        src_img = os.path.join(src_img_dir, fname)
        src_lbl = os.path.join(src_lbl_dir, fname.replace(".jpg", ".txt").replace(".png", ".txt"))

        dst_img = os.path.join(dst_img_dir, fname)
        dst_lbl = os.path.join(dst_lbl_dir, fname.replace(".jpg", ".txt").replace(".png", ".txt"))

        # Copy image
        shutil.copy2(src_img, dst_img)

        # Remap label
        if os.path.exists(src_lbl):
            remap_labels(src_lbl, dst_lbl)
        else:
            open(dst_lbl, "w").close()

def main():
    ensure_dirs()
    for split in SPLITS:
        print(f"Processing {split}...")
        process_split(split)

    print("Class remapping complete.")

if __name__ == "__main__":
    main()