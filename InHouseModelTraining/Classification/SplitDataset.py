import os
import shutil
import random

# Configuration
SAVE_ROOT = os.path.join("InHouseModelTraining", "Classification", "Images")
SRC_TRAIN = os.path.join(SAVE_ROOT, "train")
VAL_DIR = os.path.join(SAVE_ROOT, "val")
CLASSES = ["Clean", "Dirty", "NotWater"]
SPLIT_RATIO = 0.30
SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

random.seed(SEED)


def is_image_file(fn):
    return os.path.splitext(fn)[1].lower() in IMAGE_EXTS


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def unique_dest_path(dest_dir, filename):
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(dest_dir, filename)
    count = 1
    while os.path.exists(candidate):
        candidate = os.path.join(dest_dir, f"{base}_{count}{ext}")
        count += 1
    return candidate


def list_image_files(dir_path):
    if not os.path.isdir(dir_path):
        return []
    return [
        f
        for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f)) and is_image_file(f)
    ]


def split_class(src_class_dir, dest_class_dir, ratio):
    # Count images in both train and val for this class
    train_files = list_image_files(src_class_dir)
    val_files = list_image_files(dest_class_dir)

    total = len(train_files) + len(val_files)
    if total == 0:
        print(f"No image files found for class in {src_class_dir} and {dest_class_dir}")
        return 0, 0

    # Desired number of files in val to achieve the global split ratio
    desired_val = int(round(total * ratio))

    current_val = len(val_files)

    if current_val >= desired_val:
        # Already at or above desired split; do not move more from train
        print(
            f"Class directory '{os.path.basename(src_class_dir)}': "
            f"total={total}, desired_val={desired_val}, current_val={current_val} -> no move needed"
        )
        return 0, len(train_files)

    need_to_move = desired_val - current_val
    if need_to_move <= 0:
        return 0, len(train_files)

    if not train_files:
        print(f"No train image files found in {src_class_dir} to move")
        return 0, 0

    to_move_count = min(need_to_move, len(train_files))

    random.shuffle(train_files)
    to_move = train_files[:to_move_count]

    ensure_dir(dest_class_dir)
    moved = 0
    for fn in to_move:
        src_path = os.path.join(src_class_dir, fn)
        dest_path = unique_dest_path(dest_class_dir, fn)
        shutil.move(src_path, dest_path)
        moved += 1

    remaining_in_train = len(train_files) - moved
    return moved, remaining_in_train


def main():
    total_moved = 0
    total_remaining = 0

    for cls in CLASSES:
        src_dir = os.path.join(SRC_TRAIN, cls)
        dest_dir = os.path.join(VAL_DIR, cls)
        moved, remaining = split_class(src_dir, dest_dir, SPLIT_RATIO)
        total_moved += moved
        total_remaining += remaining
        print(f"Class '{cls}': moved {moved}, remaining {remaining}")

    print(f"Total moved to val: {total_moved}")
    print(f"Total remaining in train (selected classes): {total_remaining}")


if __name__ == "__main__":
    main()
