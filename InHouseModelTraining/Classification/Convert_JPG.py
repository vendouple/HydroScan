import os
import sys
from pathlib import Path
from PIL import Image
import argparse


SAVE_ROOT = os.path.join("InHouseModelTraining", "Classification", "Images", "Unsorted")
SUBFOLDERS = ("Clean", "Dirty")
SOURCE_EXTS = (
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".gif",
    ".jpeg",
    ".avif",
)
JPEG_EXTS = (".jpg", ".jpeg")


def convert_file(p: Path, out_path: Path, overwrite: bool, quality: int):
    # if output already exists and we're not overwriting, skip early
    if out_path.exists() and not overwrite:
        print(f"Skipping (exists): {out_path}")
        return False

    try:
        with Image.open(p) as im:
            # For formats with alpha, convert to RGB on white background
            if im.mode in ("RGBA", "LA") or (
                im.mode == "P" and "transparency" in im.info
            ):
                bg = Image.new("RGB", im.size, (255, 255, 255))
                im = im.convert("RGBA")
                bg.paste(im, mask=im.split()[-1])
                im = bg
            else:
                im = im.convert("RGB")

            out_path.parent.mkdir(parents=True, exist_ok=True)
            im.save(out_path, format="JPEG", quality=quality)
            return True
    except Exception as e:
        print(f"Failed to convert {p} with Pillow: {e}")

    if p.suffix.lower() == ".avif":
        try:
            import subprocess

            out_path.parent.mkdir(parents=True, exist_ok=True)
            q = max(2, min(31, int((100 - quality) / 3) + 2))
            cmd = ["ffmpeg", "-y", "-i", str(p), "-qscale:v", str(q), str(out_path)]
            subprocess.check_call(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return True
        except Exception as e2:
            print(f"FFmpeg fallback failed for {p}: {e2}")

    return False


def main():
    ap = argparse.ArgumentParser(
        description="Convert images to .jpg in Clean and Dirty folders."
    )
    ap.add_argument(
        "--remove-original",
        action="store_true",
        help="Delete original file after successful conversion.",
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing .jpg files."
    )
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality (1-100).")
    args = ap.parse_args()

    root = Path(os.path.abspath(SAVE_ROOT))
    if not root.exists():
        print(f"Root not found: {root}")
        sys.exit(1)

    total = 0
    converted = 0

    for sub in SUBFOLDERS:
        folder = root / sub
        if not folder.exists():
            print(f"Missing folder: {folder} (skipping)")
            continue

        for p in folder.rglob("*"):
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            # Skip already .jpg/.jpeg unless user asked to overwrite and ext is .jpeg (normalize to .jpg)
            if ext in JPEG_EXTS:
                # if .jpeg, optionally normalize to .jpg by saving as .jpg
                if ext == ".jpeg":
                    total += 1
                    out_path = p.with_suffix(".jpg")
                    ok = convert_file(p, out_path, args.overwrite, args.quality)
                    if ok:
                        converted += 1
                        if args.remove_original:
                            try:
                                p.unlink()
                            except Exception as e:
                                print(f"Could not remove original {p}: {e}")
                continue

            if ext in SOURCE_EXTS:
                total += 1
                out_path = p.with_suffix(".jpg")
                ok = convert_file(p, out_path, args.overwrite, args.quality)
                if ok:
                    converted += 1
                    if args.remove_original:
                        try:
                            p.unlink()
                        except Exception as e:
                            print(f"Could not remove original {p}: {e}")

    print(f"Done. Found: {total}, Converted: {converted}")


if __name__ == "__main__":
    main()
