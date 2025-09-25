import os
import sys
from pathlib import Path


PLACES365_PROTOTXT_URL = os.environ.get(
    "PLACES365_PROTOTXT_URL",
    "https://raw.githubusercontent.com/CSAILVision/places365/master/deploy_resnet152_places365.prototxt",
)
PLACES365_CAFFE_URL = os.environ.get(
    "PLACES365_CAFFE_URL",
    "http://places2.csail.mit.edu/models_places365/resnet152_places365.caffemodel",
)
PLACES365_CATEGORIES_URL = os.environ.get(
    "PLACES365_CATEGORIES_URL",
    "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt",
)
PLACES365_IO_URL = os.environ.get(
    "PLACES365_IO_URL",
    "https://raw.githubusercontent.com/CSAILVision/places365/master/IO_places365.txt",
)


def ensure_models(base_dir: str) -> None:
    """
    Ensure required model assets exist. If download fails, print concise manual instructions.

    base_dir: path to WebInterface/backend/Models
    """
    models_dir = Path(base_dir)
    place365_dir = models_dir / "Place365"
    place365_dir.mkdir(parents=True, exist_ok=True)

    prototxt_path = place365_dir / "deploy_resnet152_places365.prototxt"
    caffemodel_path = place365_dir / "resnet152_places365.caffemodel"
    categories_path = place365_dir / "categories_places365.txt"
    io_path = place365_dir / "IO_places365.txt"

    # Lazy import requests to avoid hard dependency during cold imports
    def _download_stream(
        url: str, dst: Path, label: str, chunk_size: int = 1024 * 1024
    ) -> bool:
        try:
            import requests  # type: ignore
        except Exception:
            print(f"[HydroScan] requests not available; skip downloading {label}")
            return False

        tmp = dst.with_suffix(dst.suffix + ".part")

        def _progress(bytes_read: int, total: int | None) -> None:
            if total and total > 0:
                pct = (bytes_read / total) * 100.0
                total_mb = total / (1024 * 1024)
                read_mb = bytes_read / (1024 * 1024)
                sys.stdout.write(
                    f"\r[HydroScan] Downloading {label}: {read_mb:6.1f} MB / {total_mb:6.1f} MB ({pct:5.1f}%)"
                )
            else:
                read_mb = bytes_read / (1024 * 1024)
                sys.stdout.write(
                    f"\r[HydroScan] Downloading {label}: {read_mb:6.1f} MB"
                )
            sys.stdout.flush()

        try:
            for attempt in range(3):
                try:
                    with requests.get(
                        url, stream=True, timeout=(15, 60)
                    ) as r:  # connect, read
                        r.raise_for_status()
                        total = int(r.headers.get("Content-Length", "0")) or None
                        bytes_read = 0
                        with open(tmp, "wb") as f:
                            for chunk in r.iter_content(chunk_size=chunk_size):
                                if chunk:
                                    f.write(chunk)
                                    bytes_read += len(chunk)
                                    _progress(bytes_read, total)
                    # finalize
                    if sys.stdout:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                    tmp.replace(dst)
                    print(f"[HydroScan] Downloaded {label} -> {dst}")
                    return True
                except KeyboardInterrupt:
                    try:
                        if tmp.exists():
                            tmp.unlink(missing_ok=True)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    print(f"\n[HydroScan] Download interrupted: {label}")
                    return False
                except Exception as e:
                    if attempt == 2:
                        print(f"\n[HydroScan] Failed to download {label}: {e}")
                        return False
                    else:
                        print(
                            f"\n[HydroScan] Retry downloading {label} (attempt {attempt+2}/3)"
                        )
            return False
        finally:
            try:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def _ensure(url: str, target: Path, label: str) -> None:
        if target.exists():
            return
        ok = _download_stream(url, target, label=label)
        if not ok:
            print(
                f"[HydroScan] Could not download {label.lower()} automatically. Please download manually:\n"
                f"  {url}\n"
                f"and place it at: {target}"
            )

    _ensure(PLACES365_PROTOTXT_URL, prototxt_path, "Places365 prototxt")
    _ensure(PLACES365_CAFFE_URL, caffemodel_path, "Places365 caffemodel")
    _ensure(PLACES365_CATEGORIES_URL, categories_path, "Places365 categories")
    _ensure(PLACES365_IO_URL, io_path, "Places365 IO list")
