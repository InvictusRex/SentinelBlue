import os
import shutil

SRC_ROOT = "../../Datasets/jet ski detection/train"
DST_ROOT = "../../Datasets/SentinelBlue/train"

SRC_IMG_DIR = os.path.join(SRC_ROOT, "images")
SRC_LBL_DIR = os.path.join(SRC_ROOT, "labels")

DST_IMG_DIR = os.path.join(DST_ROOT, "images")
DST_LBL_DIR = os.path.join(DST_ROOT, "labels")

OLD_CLASS_ID = 0  # jet-ski
NEW_CLASS_ID = 2  # jetski (SentinelBlue)

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

os.makedirs(DST_IMG_DIR, exist_ok=True)
os.makedirs(DST_LBL_DIR, exist_ok=True)

def get_next_index():
    existing = [
        int(os.path.splitext(f)[0])
        for f in os.listdir(DST_IMG_DIR)
        if f.split(".")[0].isdigit()
    ]
    return max(existing) + 1 if existing else 1

def remap_label(src_label_path, dst_label_path):
    new_lines = []
    with open(src_label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            parts[0] = str(NEW_CLASS_ID)
            new_lines.append(" ".join(parts))

    with open(dst_label_path, "w") as f:
        f.write("\n".join(new_lines))

def main():
    idx = get_next_index()

    for img_name in os.listdir(SRC_IMG_DIR):
        if not img_name.lower().endswith(IMAGE_EXTS):
            continue

        base = os.path.splitext(img_name)[0]
        src_img = os.path.join(SRC_IMG_DIR, img_name)
        src_lbl = os.path.join(SRC_LBL_DIR, base + ".txt")

        new_name = f"{idx:06d}"
        dst_img = os.path.join(DST_IMG_DIR, new_name + os.path.splitext(img_name)[1])
        dst_lbl = os.path.join(DST_LBL_DIR, new_name + ".txt")

        shutil.copy2(src_img, dst_img)

        if os.path.exists(src_lbl):
            remap_label(src_lbl, dst_lbl)
        else:
            open(dst_lbl, "w").close()

        idx += 1

    print("Jet-ski dataset merged with numeric filenames.")

if __name__ == "__main__":
    main()