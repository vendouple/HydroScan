import os
import shutil
import argparse
import random
import re
from typing import List


SAVE_ROOT = os.path.join("InHouseModelTraining", "Classification", "Images")
CLASS_NAMES = ["Clean", "Dirty", "NotWater"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in IMG_EXTS


def natural_key(s: str):
    parts = re.split(r"(\d+)", s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def ensure_dirs(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files = [
        f
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and is_image(f)
    ]
    files.sort(key=natural_key)
    return files


def renumber_folder(folder: str, dry_run: bool = False):
    files = list_images(folder)
    if not files:
        return
    tmp_names = []
    # Step 1: rename to temporary unique names to avoid collisions
    for idx, fname in enumerate(files, start=1):
        src = os.path.join(folder, fname)
        ext = os.path.splitext(fname)[1].lower()
        tmp = os.path.join(folder, f"__tmp_ren_{idx}{ext}")
        tmp_names.append((tmp, ext))
        if dry_run:
            print(f"[DRY] rename: {src} -> {tmp}")
        else:
            os.rename(src, tmp)
    # Step 2: rename temporaries to final sequential names
    for idx, (tmp, ext) in enumerate(tmp_names, start=1):
        final_name = f"{idx}{ext}"
        final_path = os.path.join(folder, final_name)
        if dry_run:
            print(f"[DRY] rename: {tmp} -> {final_path}")
        else:
            os.rename(tmp, final_path)
    print(f"Renamed {len(files)} files in {folder} -> 1..{len(files)}")


def split_class_folder(
    src_folder: str,
    dst_root: str,
    class_name: str,
    split_ratio: float,
    dry_run: bool = False,
):
    files = list_images(src_folder)
    if not files:
        return
    random.shuffle(files)
    split_index = int(len(files) * split_ratio)
    train_files = files[:split_index]
    val_files = files[split_index:]

    train_dir = os.path.join(dst_root, "train", class_name)
    val_dir = os.path.join(dst_root, "val", class_name)
    ensure_dirs(train_dir)
    ensure_dirs(val_dir)

    for f in train_files:
        src = os.path.join(src_folder, f)
        dst = os.path.join(train_dir, f)
        if dry_run:
            print(f"[DRY] move: {src} -> {dst}")
        else:
            shutil.move(src, dst)
    for f in val_files:
        src = os.path.join(src_folder, f)
        dst = os.path.join(val_dir, f)
        if dry_run:
            print(f"[DRY] move: {src} -> {dst}")
        else:
            shutil.move(src, dst)
    print(f"Split {class_name}: {len(train_files)} train, {len(val_files)} val")


def process_existing_train_val(root: str, dry_run: bool = False):
    for subset in ("train", "val"):
        for cname in CLASS_NAMES:
            folder = os.path.join(root, subset, cname)
            if os.path.isdir(folder):
                print(f"Renumbering {folder}")
                renumber_folder(folder, dry_run=dry_run)
            else:
                print(f"Missing folder (skipping): {folder}")


def process_split_mode(root: str, split_ratio: float, dry_run: bool = False):
    # Expect class folders directly in root
    for cname in CLASS_NAMES:
        src = os.path.join(root, cname)
        if os.path.isdir(src):
            split_class_folder(src, root, cname, split_ratio, dry_run=dry_run)
        else:
            print(f"Missing source class folder (skipping): {src}")
    # After splitting, renumber destinations
    process_existing_train_val(root, dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(
        description="Image sorter/renamer for train/val classification layout."
    )
    parser.add_argument("--root", default=SAVE_ROOT, help="Root images folder")
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Train split ratio when splitting (default 0.8)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print actions without making changes"
    )
    args = parser.parse_args()

    root = args.root
    if not os.path.exists(root):
        print(f"Root does not exist: {root}")
        return

    has_train = os.path.isdir(os.path.join(root, "train"))
    has_val = os.path.isdir(os.path.join(root, "val"))
    has_classes_direct = any(os.path.isdir(os.path.join(root, c)) for c in CLASS_NAMES)

    if has_train or has_val:
        print(
            "Detected train/ or val/ folders. Renumbering existing train/val class folders."
        )
        process_existing_train_val(root, dry_run=args.dry_run)
    elif has_classes_direct:
        print(
            "Detected class folders directly under root. Splitting into train/ and val/."
        )
        process_split_mode(root, args.split_ratio, dry_run=args.dry_run)
    else:
        print("No recognizable class/train/val layout found under root.")
        print(
            f"Expected either: {os.path.join(root, 'train')} + {os.path.join(root, 'val')} or class folders {CLASS_NAMES}"
        )


if __name__ == "__main__":
    main()
