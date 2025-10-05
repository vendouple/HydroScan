import os
os.environ["USE_LIBUV"] = "0"
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path
import multiprocessing
import torch

# ============================================
# CONFIGURATION
# ============================================
HERE = Path(__file__).resolve().parent
DATA_YAML = HERE / "data_cls.yml"
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

# ============================================
# TRAINING CONFIG
# ============================================
EPOCHS = 200
BATCH_SIZE = 64
IMG_SIZE = 224
LEARNING_RATE = 0.2
LR_FINAL = 0.01
WARMUP_EPOCHS = 0
WARMUP_BIAS_LR = 0.1
WEIGHT_DECAY = 0.0001
COS_LR = True
HUE_SATURATION = 0.4
OPTIMIZER = "SGD"
DEVICE = "0"  # set to "cpu" or "0,1" for multi-GPU

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
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"DATA_YAML not found at {DATA_YAML}. Please check path.")

    print("🚀 Starting YOLOv12 Cls training...")
    torch.cuda.empty_cache()
    results = model.train(
        data="C:/Users/PC/Desktop/HydroScan/InHouseModelTraining/Classification/Images",
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        lr0=LEARNING_RATE,
        lrf=LR_FINAL,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_bias_lr=WARMUP_BIAS_LR,
        weight_decay=WEIGHT_DECAY,
        cos_lr=COS_LR,
        hsv_s=HUE_SATURATION,
        optimizer=OPTIMIZER,
        device=DEVICE,
        project=str(run_dir),
        name="classification",
        exist_ok=True,
        task="classify",  # ✅ Force classification mode
    )

    # ============================================
    # EXPORT METRICS
    # ============================================
    print("📊 Exporting training metrics...")
    metrics = getattr(results, "results_dict", None) or {}

    csv_path = run_dir / "training_metrics.csv"
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    print(f"✅ Metrics saved to {csv_path}")

    # ============================================
    # VALIDATION
    # ============================================
    print("🧪 Running validation...")
    val_results = model.val(data=str(DATA_YAML.as_posix()), imgsz=IMG_SIZE, task="classify")

    # ============================================
    # VISUALIZATION
    # ============================================
    print("📈 Plotting training graphs...")

    try:
        # Loss curve
        if hasattr(results, "tloss") and results.tloss is not None:
            plt.figure(figsize=(10, 6))
            plt.plot(results.tloss, label="Training Loss", color="blue")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss Curve (YOLOv12 Cls)")
            plt.legend()
            plt.grid(True)
            plt.savefig(run_dir / "training_loss.png")
            plt.close()

        # Accuracy/F1 curve
        if hasattr(results, "metrics") and isinstance(results.metrics, dict):
            acc = results.metrics.get("accuracy", [])
            f1 = results.metrics.get("f1", [])
            plt.figure(figsize=(10, 6))
            if acc: plt.plot(acc, label="Accuracy", color="green")
            if f1: plt.plot(f1, label="F1 Score", color="orange")
            plt.xlabel("Epoch")
            plt.ylabel("Metric Value")
            plt.title("Accuracy and F1 Curve (YOLOv12 Cls)")
            plt.legend()
            plt.grid(True)
            plt.savefig(run_dir / "metrics_curve.png")
            plt.close()
    except Exception as e:
        print(f"⚠️ Could not plot metrics: {e}")

    print(f"✅ Training completed! All results saved in:\n{run_dir}")
