from ultralytics import YOLO
import json
from pathlib import Path
import csv
import numpy as np
from PIL import Image, ImageDraw
import torch

# Minimal train call using YOLOv12-X classification checkpoint
model = YOLO("./InHouseModelTraining/Classification/yolov12x-cls.pt")

# Auto-detect device(s): prefer all available GPUs, fall back to single GPU or CPU
n_gpus = torch.cuda.device_count()
if n_gpus <= 0:
    device_str = "cpu"
    print("No CUDA GPUs detected — training will run on CPU.")
elif n_gpus == 1:
    device_str = "0"
    print("Using single GPU: cuda:0")
else:
    device_str = ",".join(str(i) for i in range(n_gpus))
    print(f"Using {n_gpus} GPUs: {device_str}")

results = None
try:
    results = model.train(
        data="Images",
        epochs=200,
        imgsz=224,
        project="results",
        name="classification_model",
        verbose=True,
        device=device_str,
    )
except Exception as e_multi:
    print(f"Training on device '{device_str}' failed: {e_multi}")
    if device_str != "cpu":
        # try single GPU if multi-GPU failed
        try:
            fallback = "0" if torch.cuda.device_count() > 0 else "cpu"
            print(f"Retrying training on '{fallback}'")
            results = model.train(
                data="Images",
                epochs=200,
                imgsz=224,
                project="results",
                name="classification_model",
                verbose=True,
                device=fallback,
            )
        except Exception as e_single:
            print(
                f"Single-GPU training also failed: {e_single}\nFalling back to CPU training."
            )
            try:
                results = model.train(
                    data="Images",
                    epochs=200,
                    imgsz=224,
                    project="results",
                    name="classification_model",
                    verbose=True,
                    device="cpu",
                )
            except Exception as e_cpu:
                print(f"CPU training failed as well: {e_cpu}")
                raise

run_base = Path("results") / "classification_model"
if not run_base.exists():
    # try alternative naming with timestamp
    runs = sorted(
        Path("results").glob("classification_model*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    run_base = runs[0] if runs else Path("results") / "classification_model"

run_base.mkdir(parents=True, exist_ok=True)

# If model.val() is available, run validation to get per-sample predictions
try:
    val_results = model.val(data="Images/val", imgsz=224, batch=16, save=False)

    preds = []
    if isinstance(val_results, dict) and "predictions" in val_results:
        preds = val_results["predictions"]
    elif hasattr(val_results, "preds"):
        preds = val_results.preds
except Exception:
    preds = []

# If preds not available, attempt a manual pass over Images/val
if not preds:
    try:
        preds = []
        val_paths = list(Path("Images").joinpath("val").rglob("*.*"))
        batch = []
        for p in val_paths:
            if not p.is_file():
                continue
            batch.append(str(p))
            if len(batch) >= 16:
                out = model(batch, imgsz=224)
                for o, fp in zip(out, batch):
                    try:
                        cls = int(o.boxes.cls[0].item()) if len(o.boxes.cls) else -1
                        prob = (
                            float(o.boxes.conf[0].item()) if len(o.boxes.conf) else 0.0
                        )
                    except Exception:
                        cls = -1
                        prob = 0.0
                    preds.append({"path": fp, "pred": cls, "prob": prob})
                batch = []
        if batch:
            out = model(batch, imgsz=224)
            for o, fp in zip(out, batch):
                try:
                    cls = int(o.boxes.cls[0].item()) if len(o.boxes.cls) else -1
                    prob = float(o.boxes.conf[0].item()) if len(o.boxes.conf) else 0.0
                except Exception:
                    cls = -1
                    prob = 0.0
                preds.append({"path": fp, "pred": cls, "prob": prob})
    except Exception:
        preds = []

# Save predictions CSV
pred_csv = run_base / "predictions.csv"
try:
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "pred", "prob"])
        for r in preds:
            w.writerow([r.get("path", ""), r.get("pred", ""), r.get("prob", "")])
except Exception as e:
    print("Failed to write predictions.csv:", e)

# Attempt to build a confusion matrix if labels are available in Images/val folder structure
val_root = Path("Images") / "val"
if val_root.exists():
    class_names = sorted([d.name for d in val_root.iterdir() if d.is_dir()])
    name_to_idx = {n: i for i, n in enumerate(class_names)}

    y_true = []
    y_pred = []
    for p in val_root.rglob("*.*"):
        if p.is_file() and p.parent.name in name_to_idx:
            y_true.append(name_to_idx[p.parent.name])
            # try to find pred for this path
            match = next(
                (x for x in preds if Path(x.get("path", "")).resolve() == p.resolve()),
                None,
            )
            if match:
                y_pred.append(match.get("pred", -1))
            else:
                y_pred.append(-1)

    if y_true and y_pred:
        cm = np.zeros((len(class_names), len(class_names)), dtype=int)
        for t, pr in zip(y_true, y_pred):
            if 0 <= t < len(class_names) and 0 <= pr < len(class_names):
                cm[t, pr] += 1

        # save confusion matrix as JSON and PNG
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            plt.colorbar()
            plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
            plt.yticks(range(len(class_names)), class_names)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(run_base / "confusion_matrix.png")
            plt.close()
        except Exception as e:
            print("Failed to plot confusion matrix:", e)

        # per-class metrics
        per_class = []
        for i, cname in enumerate(class_names):
            tp = int(cm[i, i])
            fn = int(cm[i, :].sum() - tp)
            fp = int(cm[:, i].sum() - tp)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            support = int(cm[i, :].sum())
            per_class.append(
                {
                    "class": cname,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "support": support,
                }
            )

        try:
            with open(
                run_base / "per_class_metrics.csv", "w", newline="", encoding="utf-8"
            ) as f:
                w = csv.writer(f)
                w.writerow(["class", "precision", "recall", "f1", "support"])
                for r in per_class:
                    w.writerow(
                        [
                            r["class"],
                            f"{r['precision']:.4f}",
                            f"{r['recall']:.4f}",
                            f"{r['f1']:.4f}",
                            r["support"],
                        ]
                    )
        except Exception as e:
            print("Failed to write per_class_metrics.csv:", e)

# Save a few sample prediction images with overlay
try:
    samples_dir = run_base / "sample_predictions"
    samples_dir.mkdir(parents=True, exist_ok=True)
    for r in preds[:50]:
        p = Path(r.get("path", ""))
        if not p.exists():
            continue
        try:
            img = Image.open(p).convert("RGB")
            draw = ImageDraw.Draw(img)
            txt = f"Pred: {r.get('pred','')} ({r.get('prob',0):.2f})"
            draw.rectangle([0, 0, img.width, 16], fill=(0, 0, 0))
            draw.text((2, 0), txt, fill=(255, 255, 255))
            dest = samples_dir / p.name
            img.save(dest)
        except Exception:
            continue
except Exception as e:
    print("Failed to save sample images:", e)
