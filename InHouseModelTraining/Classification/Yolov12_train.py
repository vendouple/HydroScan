import os
os.environ["USE_LIBUV"] = "0"
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import json
import ast
from pathlib import Path
import multiprocessing
import torch

# ============================================
# CONFIGURATION
# ============================================
HERE = Path(__file__).resolve().parent
DATA_YAML = HERE / "data_cls.yml"
DATA_DIR = HERE / "Images"  # Classification expects a directory, not a YAML
RESULTS_DIR = HERE / "results"
PREFERRED_WEIGHT = HERE / "yolov12x.pt"  # or yolov12x-cls.pt if that's your actual file
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============================================
# UTILITIES
# ============================================
def find_weights(directory: Path, preferred: Path | None = None) -> Path:
    """Find .pt weights in directory, preferring yolov* or yolo* names."""
    if preferred and preferred.exists():
        return preferred
    candidates = list(directory.glob("*.pt"))
    prioritized = [p for p in candidates if p.name.lower().startswith(("yolo", "yolov"))]
    if prioritized:
        return prioritized[0]
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"No .pt weights found in {directory}. Expected something like yolov12x-cls.pt"
    )


def parse_data_yaml(path: Path) -> dict:
    """Parse a simple dataset YAML file without external dependencies."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset YAML not found at {path}")

    data: dict[str, object] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            try:
                data[key] = ast.literal_eval(value)
            except Exception:
                data[key] = value
        else:
            data[key] = value.strip().strip('"\'')
    return data


def summarize_dataset(data_yaml: Path, run_dir: Path) -> dict:
    """Compute class counts per split and persist them for analysis."""
    cfg = parse_data_yaml(data_yaml)
    class_names = cfg.get("names", [])
    if isinstance(class_names, dict):  # YAML can map index->name
        class_names = list(class_names.values())
    if isinstance(class_names, str):
        class_names = [name.strip() for name in class_names.strip("[]").split(",") if name.strip()]

    summary = {"class_names": class_names}
    rows = []

    for split in ("train", "val"):
        split_path_str = cfg.get(split)
        split_path = Path(split_path_str) if split_path_str else None
        counts: dict[str, int] = {}
        if split_path and split_path.exists():
            for cls in class_names:
                cls_dir = split_path / cls
                if cls_dir.exists():
                    counts[cls] = sum(1 for item in cls_dir.iterdir() if item.is_file())
                else:
                    counts[cls] = 0
        total = sum(counts.values())
        percentages = {
            cls: round((counts.get(cls, 0) / total) * 100, 2) if total else 0.0 for cls in class_names
        }
        summary[split] = {
            "path": str(split_path) if split_path else None,
            "counts": counts,
            "total": total,
            "percentages": percentages,
        }
        for cls in class_names:
            rows.append(
                {
                    "split": split,
                    "class": cls,
                    "count": counts.get(cls, 0),
                    "percentage": percentages.get(cls, 0.0),
                }
            )

    train_info = summary.get("train") if isinstance(summary.get("train"), dict) else None
    train_counts = train_info.get("counts", {}) if isinstance(train_info, dict) else {}
    total_train = train_info.get("total", 0) if isinstance(train_info, dict) else 0
    suggested_weights = {
        cls: round(total_train / count, 4) if count else 0.0 for cls, count in train_counts.items()
    } if total_train else {}
    summary["suggested_class_weights"] = suggested_weights

    # Persist summaries for later review
    (run_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if rows:
        pd.DataFrame(rows).to_csv(run_dir / "dataset_summary.csv", index=False)

    print("📦 Dataset snapshot:")
    for split in ("train", "val"):
        info = summary.get(split)
        if not isinstance(info, dict):
            continue
        details = ", ".join(
            f"{cls}={info['counts'].get(cls, 0)} ({info['percentages'].get(cls, 0.0)}%)"
            for cls in class_names
        )
        print(f"  • {split}: {info.get('total', 0)} images -> {details}")
    if suggested_weights:
        print(f"  • Suggested class weights (inverse frequency): {suggested_weights}")

    return summary

# ============================================
# TRAINING CONFIG
# ============================================
EPOCHS = 80
BATCH_SIZE = 32
IMG_SIZE = 320  
LEARNING_RATE = 5e-4
LR_FINAL = 0.05
WARMUP_EPOCHS = 5
WEIGHT_DECAY = 5e-4
COS_LR = True
OPTIMIZER = "AdamW"
PATIENCE = 20
DROPOUT = 0.2
MIXUP = 0.2
CUTMIX = 0.1
ERASING = 0.4
SEED = 42
WORKERS = max(1, min(8, os.cpu_count() or 1))
DEVICE = "0" if torch.cuda.is_available() else "cpu"

# ============================================
# MAIN EXECUTION (Windows-safe)
# ============================================
if __name__ == "__main__":
    multiprocessing.freeze_support()

    # ============================================
    # SETUP
    # ============================================
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_summary = summarize_dataset(DATA_YAML, run_dir)
    class_names = dataset_summary.get("class_names", [])
    suggested_weights = dataset_summary.get("suggested_class_weights", {})

    # ============================================
    # LOAD MODEL (Force classification mode)
    # ============================================
    try:
        weights_path = find_weights(HERE, preferred=PREFERRED_WEIGHT)
    except FileNotFoundError as e:
        files = sorted(p.name for p in HERE.iterdir())
        raise FileNotFoundError(f"Could not locate weights. {e}\nFiles: {files}")

    print(f"🔍 Loading YOLOv12 (Cls) model from {weights_path}")

    # ✅ Force classification task
    model = YOLO(str(weights_path.as_posix()), task="classify")

    # ============================================
    # TRAIN MODEL
    # ============================================
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR not found at {DATA_DIR}. Please check path.")

    print("🚀 Starting YOLOv12 Cls training...")
    torch.cuda.empty_cache()
    results = model.train(
        data=str(DATA_DIR),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        lr0=LEARNING_RATE,
        lrf=LR_FINAL,
        warmup_epochs=WARMUP_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        cos_lr=COS_LR,
        optimizer=OPTIMIZER,
        patience=PATIENCE,
        dropout=DROPOUT,
        mixup=MIXUP,
        cutmix=CUTMIX,
        erasing=ERASING,
        amp=True,  # Enable automatic mixed precision for memory savings
        val=True,
        plots=True,
        seed=SEED,
        device=DEVICE,
        workers=WORKERS,
        project=str(run_dir),
        name="classification",
        exist_ok=True,
        task="classify",
    )

    # ============================================
    # EXPORT METRICS
    # ============================================
    print("📊 Exporting training metrics...")
    history_df = None
    history_csv = Path(results.save_dir) / "results.csv"
    if history_csv.exists():
        history_df = pd.read_csv(history_csv)
        history_df.to_csv(run_dir / "metrics_history.csv", index=False)
        print(f"✅ Saved epoch history to {run_dir / 'metrics_history.csv'}")
    else:
        print(f"⚠️ {history_csv.name} not found in {results.save_dir}")

    def safe_number(value):
        if pd.isna(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return value

    final_metrics = {}
    if history_df is not None and not history_df.empty:
        final_row = history_df.iloc[-1].to_dict()
        final_metrics = {
            key: safe_number(value)
            for key, value in final_row.items()
            if key not in {"epoch"}
        }

        top1_cols = [c for c in history_df.columns if "accuracy_top1" in c]
        if top1_cols:
            top1_col = top1_cols[0]
            top1_series = history_df[top1_col]
            if not top1_series.empty:
                best_idx = int(top1_series.idxmax())
                best_val = safe_number(top1_series.loc[best_idx])
                best_epoch = safe_number(history_df.loc[best_idx, "epoch"])
                final_metrics["best_top1_epoch"] = best_epoch
                final_metrics["best_top1"] = best_val
    else:
        final_metrics = getattr(results, "results_dict", None) or {}

    metrics_json = run_dir / "training_metrics.json"
    metrics_csv = run_dir / "training_metrics.csv"
    metrics_json.write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    pd.DataFrame([final_metrics]).to_csv(metrics_csv, index=False)
    print(f"✅ Metrics saved to {metrics_json} and {metrics_csv}")

    # ============================================
    # VALIDATION
    # ============================================
    print("🧪 Running validation...")
    val_results = model.val(data=str(DATA_DIR.as_posix()), imgsz=IMG_SIZE, task="classify")

    # ============================================
    # VISUALIZATION
    # ============================================
    print("📈 Plotting training graphs...")

    try:
        history_file = run_dir / "metrics_history.csv"
        history_df = None
        if history_file.exists():
            history_df = pd.read_csv(history_file)

        if history_df is not None and not history_df.empty:
            plots_dir = run_dir / "plots"
            plots_dir.mkdir(exist_ok=True)

            # Loss curves
            for column in ["train/loss", "val/loss"]:
                if column in history_df.columns:
                    plt.figure(figsize=(10, 6))
                    plt.plot(history_df["epoch"], history_df[column], label=column)
                    plt.xlabel("Epoch")
                    plt.ylabel(column)
                    plt.title(f"{column} over epochs")
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(plots_dir / f"{column.replace('/', '_')}.png")
                    plt.close()

            metric_columns = [c for c in history_df.columns if c.startswith("accuracy") or c.startswith("top")]
            if metric_columns:
                plt.figure(figsize=(10, 6))
                for column in metric_columns:
                    plt.plot(history_df["epoch"], history_df[column], label=column)
                plt.xlabel("Epoch")
                plt.ylabel("Metric value")
                plt.title("Accuracy metrics over epochs")
                plt.legend()
                plt.grid(True)
                plt.savefig(plots_dir / "accuracy_metrics.png")
                plt.close()

            cm_json = Path(results.save_dir) / "confusion_matrix.json"
            if cm_json.exists():
                cm_data = json.loads(cm_json.read_text(encoding="utf-8"))
                if isinstance(cm_data, dict) and "matrix" in cm_data:
                    matrix = cm_data.get("matrix")
                    if matrix:
                        cm_df = pd.DataFrame(matrix, index=class_names or None, columns=class_names or None)
                        cm_df.to_csv(run_dir / "confusion_matrix.csv")

                        plt.figure(figsize=(8, 6))
                        plt.imshow(cm_df, interpolation="nearest", cmap="Blues")
                        plt.title("Confusion matrix (counts)")
                        plt.colorbar()
                        tick_marks = range(len(cm_df.columns))
                        plt.xticks(tick_marks, list(cm_df.columns), rotation=45, ha="right")
                        plt.yticks(tick_marks, list(cm_df.index))
                        plt.tight_layout()
                        plt.ylabel("True label")
                        plt.xlabel("Predicted label")
                        plt.savefig(plots_dir / "confusion_matrix_counts.png")
                        plt.close()

            if class_names and "val_metrics/class_accuracy" in history_df.columns:
                class_acc_col = "val_metrics/class_accuracy"
                per_class = history_df[class_acc_col].dropna()
                if not per_class.empty:
                    last_row = per_class.iloc[-1]
                    if isinstance(last_row, str) and last_row.startswith("["):
                        try:
                            values = ast.literal_eval(last_row)
                            plt.figure(figsize=(8, 6))
                            plt.bar(class_names, values)
                            plt.ylim(0, 1)
                            plt.ylabel("Accuracy")
                            plt.title("Validation accuracy per class")
                            plt.grid(axis="y", linestyle="--", alpha=0.4)
                            plt.savefig(plots_dir / "val_accuracy_per_class.png")
                            plt.close()
                        except Exception as e:
                            print(f"⚠️ Could not parse class accuracy array: {e}")

    except Exception as e:
        print(f"⚠️ Could not plot metrics: {e}")

    print(f"✅ Training completed! All results saved in:\n{run_dir}")
