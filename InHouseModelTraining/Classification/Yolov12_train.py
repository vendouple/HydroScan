#!/usr/bin/env python3
"""
Yolov12_train.py — dataset-dir & AMP-safe full update

- ROOT_DIR defaults to script folder.
- Passes dataset folder (Images/) to Ultralytics classify trainer (expects train/val subfolders).
- AMP (amp=True) is used only when GPUs are present.
- No CLI args; edit CONFIG at top.
"""

from pathlib import Path
import random
import time
import sys
import os
import inspect

# ------------- CONFIG: edit these variables -------------
ROOT_DIR = Path(__file__).resolve().parent  # script directory
WEIGHTS = "yolov12x-cls.pt"
EPOCHS = 80
IMG_SZ = 640
BATCH_SIZE = 16
LR = 0.01
NAME = None
RESULTS_DIR = "results"
AMP = True  # request AMP (only used automatically when GPU present)
WORKERS = 8
DEVICE = None  # None -> auto detect GPUs (or "cpu" to force CPU, or "0,1" etc)
NUM_SAMPLE_PRED = 16
EXIST_OK = False
# -------------------------------------------------------

import torch
from ultralytics import YOLO

# optional plotting helper from ultralytics
try:
    from ultralytics.utils.plotting import plot_results
except Exception:
    plot_results = None


def find_weights(root: Path, weights_name: str):
    candidate = root / weights_name
    if candidate.exists():
        print(f"[INFO] Found weights at: {candidate}")
        return candidate.resolve()
    patterns = ["yolov12*.pt", "yolov12*cls*.pt", "*yolov12*.pt", "*cls*.pt", "*.pt"]
    print(
        f"[WARN] Exact weights not found at {candidate}. Searching patterns under {root} ..."
    )
    for pat in patterns:
        found = list(root.rglob(pat))
        if found:
            found_sorted = sorted(
                found, key=lambda p: (len(p.parts), -p.stat().st_mtime)
            )
            chosen = found_sorted[0]
            print(f"[INFO] Found candidate weights by pattern '{pat}': {chosen}")
            return chosen.resolve()
    return None


def fallback_plot_results(csv_path: Path, out_png: Path):
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except Exception:
        return False
    try:
        df = pd.read_csv(csv_path)
        numeric = df.select_dtypes(include="number").columns.tolist()
        cols = numeric[:3]
        if not cols:
            return False
        plt.figure(figsize=(8, 5))
        for c in cols:
            plt.plot(df.index, df[c], label=c)
        plt.xlabel("step/index")
        plt.ylabel("value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(out_png))
        plt.close()
        return True
    except Exception:
        return False


def build_data_yaml(root: Path, out_yaml: Path):
    """
    Build a bookkeeping data yaml (not required by trainer when passing dataset dir).
    """
    # look for train/val with common capitalizations
    train_dir = root / "Images" / "train"
    val_dir = root / "Images" / "val"
    # fallback capitalization
    if not train_dir.exists():
        train_dir = root / "Images" / "Train"
    if not val_dir.exists():
        val_dir = root / "Images" / "Val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Expected train/val under {root / 'Images'} (looked for {train_dir} and {val_dir})"
        )
    classes = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    if len(classes) == 0:
        raise RuntimeError(f"No class subfolders found in {train_dir}")
    yaml_text = (
        f"names: {classes}\n"
        f"nc: {len(classes)}\n"
        f"train: {str(train_dir.resolve())}\n"
        f"val: {str(val_dir.resolve())}\n"
    )
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml_text)
    print(f"[INFO] Built data yaml with {len(classes)} classes -> {out_yaml}")
    return out_yaml


def sample_val_images(val_dir: Path, n: int = 10):
    # accept 'val' or 'Val'
    if not val_dir.exists():
        alt = (
            val_dir.parent / "Val"
            if val_dir.name.lower() == "val"
            else val_dir.parent / "val"
        )
        if alt.exists():
            val_dir = alt
    imgs = []
    for cls in val_dir.iterdir():
        if cls.is_dir():
            imgs.extend(
                [
                    p
                    for p in cls.rglob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                ]
            )
    if not imgs:
        raise RuntimeError(f"No images found under {val_dir}")
    random.shuffle(imgs)
    return imgs[:n]


def newest_run_dir(project_dir: Path, name: str = None):
    if not project_dir.exists():
        return None
    cand = [p for p in project_dir.iterdir() if p.is_dir()]
    # include nested dirs one level deep (ultralytics variant)
    for p in list(project_dir.iterdir()):
        if p.is_dir():
            cand.extend([q for q in p.iterdir() if q.is_dir()])
    if name:
        cand = [p for p in cand if name in str(p)]
    if not cand:
        return None
    cand_sorted = sorted(cand, key=lambda p: p.stat().st_mtime, reverse=True)
    return cand_sorted[0]


def find_train_val_dirs(images_root: Path):
    """
    Return the dataset_dir (Images/) and resolved train/val dirs (case-insensitive).
    Raises helpful errors if missing.
    """
    dataset_dir = images_root
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")
    # resolve train
    train_candidates = [dataset_dir / "train", dataset_dir / "Train"]
    val_candidates = [dataset_dir / "val", dataset_dir / "Val"]
    train_dir = next((p for p in train_candidates if p.exists()), None)
    val_dir = next((p for p in val_candidates if p.exists()), None)
    if train_dir is None:
        raise FileNotFoundError(
            f"No 'train' folder found under {dataset_dir}. Expected one of: {[str(p) for p in train_candidates]}"
        )
    if val_dir is None:
        raise FileNotFoundError(
            f"No 'val' folder found under {dataset_dir}. Expected one of: {[str(p) for p in val_candidates]}"
        )
    # check class subfolders
    classes = [p for p in train_dir.iterdir() if p.is_dir()]
    if not classes:
        raise FileNotFoundError(
            f"No class subfolders found in {train_dir}. Expected class folders (one per class)."
        )
    return dataset_dir, train_dir, val_dir


def print_train_signature(model):
    try:
        sig = inspect.signature(model.train)
        print("[INFO] model.train signature:", sig)
    except Exception:
        print("[WARN] Could not inspect model.train signature.")


def main():
    print(f"[INFO] Script location (ROOT_DIR): {ROOT_DIR}")

    weights_path = find_weights(ROOT_DIR, WEIGHTS)
    if weights_path is None:
        print("[ERROR] No weights file found. Listing files for debugging:")
        for p in sorted(ROOT_DIR.iterdir()):
            print("  -", p.name)
        raise FileNotFoundError(
            f"Weights file not found. Expected '{WEIGHTS}' in {ROOT_DIR} or similar."
        )

    # Build data yaml for bookkeeping (not used directly by trainer below)
    data_yaml = ROOT_DIR / "data_cls.yaml"
    build_data_yaml(ROOT_DIR, data_yaml)

    # decide device
    if DEVICE:
        device = DEVICE
    else:
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            device = "cpu"
        elif n_gpus == 1:
            device = "0"
        else:
            device = ",".join(str(i) for i in range(n_gpus))
    print(f"[INFO] Using device: {device}")

    use_amp = AMP and (device != "cpu")
    if AMP and device == "cpu":
        print(
            "[WARN] AMP requested but running on CPU — AMP will be disabled (no fp16 on CPU)."
        )

    project_dir = ROOT_DIR / RESULTS_DIR
    project_dir.mkdir(parents=True, exist_ok=True)

    run_name = NAME or f"y12x_cls_{int(time.time())}"
    print(f"[INFO] Run name: {run_name}")
    print(f"[INFO] Outputs will be saved under: {project_dir}")

    # initialize model
    model = YOLO(str(weights_path))

    # Resolve dataset dir and ensure train/val/class structure exists:
    images_root = ROOT_DIR / "Images"
    try:
        dataset_dir, train_dir, val_dir = find_train_val_dirs(images_root)
    except Exception as e:
        print("[ERROR] Dataset layout problem:", e)
        raise

    print(f"[INFO] Trainer will be pointed at dataset dir: {dataset_dir}")
    print(f"[INFO] Train dir: {train_dir}")
    print(f"[INFO] Val dir:   {val_dir}")

    # Build train kwargs: pass dataset_dir (not data yaml)
    train_kwargs = dict(
        task="classify",
        data=str(dataset_dir),  # IMPORTANT: pass folder that contains train/ & val/
        epochs=EPOCHS,
        imgsz=IMG_SZ,
        batch=BATCH_SIZE,
        lr0=LR,
        device=device,
        project=str(project_dir),
        name=run_name,
        exist_ok=EXIST_OK,
        workers=WORKERS,
    )

    if use_amp:
        train_kwargs["amp"] = True
        print("[INFO] AMP will be enabled for training (amp=True).")
    else:
        print("[INFO] AMP not enabled for this run.")

    print("[INFO] Starting training with parameters:")
    for k, v in train_kwargs.items():
        print(f"  {k}: {v}")

    # Try to train and provide helpful debugging if ultralytics complains about kwargs
    try:
        results = model.train(**train_kwargs)
        print("[INFO] Training finished. model.train() returned:", type(results))
    except TypeError as e:
        # often caused by an unexpected kwarg for this ultralytics version
        print(
            "[ERROR] TypeError when calling model.train(). This often means a kwarg is unsupported for your ultralytics version."
        )
        print("Exception:", e)
        print_train_signature(model)
        raise
    except Exception as e:
        print("[ERROR] model.train() raised an exception:")
        raise

    # locate run directory
    run_dir = newest_run_dir(project_dir, name=run_name)
    if run_dir is None:
        run_dir = project_dir / run_name
    print("[INFO] Resolved run dir:", run_dir)

    # create results plot if possible
    results_csv = run_dir / "results.csv"
    out_png = run_dir / "results.png"
    if results_csv.exists():
        plotted = False
        if plot_results:
            try:
                print("[INFO] Using ultralytics.plot_results to create results.png")
                plot_results(file=str(results_csv), dir=str(run_dir))
                plotted = True
            except Exception as e:
                print("[WARN] ultralytics.plot_results failed:", e)
        if not plotted:
            ok = fallback_plot_results(results_csv, out_png)
            if ok:
                print("[INFO] Saved fallback results plot to:", out_png)
            else:
                print(
                    "[WARN] Could not create results plot (no suitable plot helper found)."
                )
    else:
        print("[WARN] results.csv not found in run folder — skipping plotting.")

    # sample val predictions
    try:
        sample_imgs = sample_val_images(val_dir, n=NUM_SAMPLE_PRED)
        samples_out = run_dir / "samples"
        samples_out.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Running predictions on {len(sample_imgs)} sample val images ...")
        for i, img in enumerate(sample_imgs):
            model.predict(
                source=str(img),
                save=True,
                project=str(samples_out),
                name=f"sample_{i}",
                exist_ok=True,
            )
        print("[INFO] Sample predictions saved under:", samples_out)
    except Exception as e:
        print("[WARN] Failed to save sample predictions:", e)

    print(
        "[DONE] Check the run directory for weights (best.pt/last.pt), logs, results.csv, results.png and samples."
    )


if __name__ == "__main__":
    main()
