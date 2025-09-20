from __future__ import annotations
import os
import io
import pickle
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
from loguru import logger

PLACES_MODEL_URL = (
    "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"
)
CATEGORIES_URL = "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt"
IO_URL = (
    "https://raw.githubusercontent.com/CSAILVision/places365/master/IO_places365.txt"
)


class Places365SceneClassifier:
    def __init__(
        self,
        cache_dir: str = "./data/cache",
        skip_download: bool = False,
        device: str = "cpu",
    ):
        self.cache_dir = cache_dir
        self.device = device
        os.makedirs(self.cache_dir, exist_ok=True)
        self.model_path = os.path.join(cache_dir, "resnet18_places365.pth.tar")
        self.categories_path = os.path.join(cache_dir, "categories_places365.txt")
        self.io_path = os.path.join(cache_dir, "IO_places365.txt")
        if not skip_download:
            self._ensure_assets()
        self._load()

    def _download(self, url: str, path: str):
        if os.path.exists(path):
            return
        logger.info(f"Downloading {url} -> {path}")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)

    def _ensure_assets(self):
        try:
            self._download(PLACES_MODEL_URL, self.model_path)
        except Exception as e:
            logger.warning(f"Failed to download model: {e}")
        try:
            self._download(CATEGORIES_URL, self.categories_path)
            self._download(IO_URL, self.io_path)
        except Exception as e:
            logger.warning(f"Failed to download metadata: {e}")

    def _load(self):
        # Build a resnet18 and load places365 weights
        self.model = models.resnet18(num_classes=365)
        if os.path.exists(self.model_path):
            # handle python2 pickles
            try:
                from functools import partial
                import pickle as pkl

                pkl.load = partial(pkl.load, encoding="latin1")
                pkl.Unpickler = partial(pkl.Unpickler, encoding="latin1")
                checkpoint = torch.load(
                    self.model_path, map_location=self.device, pickle_module=pkl
                )
            except Exception:
                checkpoint = torch.load(self.model_path, map_location=self.device)
            state_dict = (
                checkpoint["state_dict"]
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint
                else checkpoint
            )
            # Strip 'module.' prefix if present
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace("module.", "")
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()
        self.model.to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        # categories
        self.categories = []
        if os.path.exists(self.categories_path):
            with open(self.categories_path, "r") as f:
                for line in f:
                    if line.strip():
                        cat = line.strip().split(" ")[0][3:]
                        self.categories.append(cat)
        # IO mapping: 0 indoor, 1 outdoor
        self.io_map = {}
        if os.path.exists(self.io_path):
            with open(self.io_path, "r") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        label, val = line.split(" ")
                        self.io_map[int(val)] = label
                    except Exception:
                        # file format can be different; fallback to indices list of outdoor
                        pass

    def classify(self, img_rgb) -> dict:
        # img_rgb: numpy array HxWx3
        image = Image.fromarray(img_rgb)
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
            top5_prob, top5_catid = torch.topk(probs, 5)
        top = [
            (
                float(top5_prob[i].item()),
                (
                    self.categories[top5_catid[i]]
                    if self.categories
                    else str(int(top5_catid[i].item()))
                ),
            )
            for i in range(5)
        ]
        # crude indoor/outdoor heuristic: if any top category contains outdoor words
        scene = "Indoor/Unknown"
        conf = float(top[0][0]) if top else 0.0
        if top:
            label = top[0][1]
            outdoor_words = [
                "outdoor",
                "beach",
                "forest",
                "river",
                "lake",
                "mountain",
                "park",
                "field",
                "coast",
            ]
            if any(w in label for w in outdoor_words):
                scene = "Outdoor/Natural"
        return {"scene": scene, "confidence": conf, "top5": top}

    def info(self) -> dict:
        return {
            "name": "Places365-ResNet18",
            "device": self.device,
            "assets": {
                "model": os.path.exists(self.model_path),
                "categories": os.path.exists(self.categories_path),
            },
        }
