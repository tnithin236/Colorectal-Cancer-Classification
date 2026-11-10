"""Loader matched to the training notebook.

Architecture: torchvision EfficientNet-B0, classifier = [Dropout(0.2), Linear(1280, 2)]
Checkpoints:
  best_colorectal_model.pt            -> raw state_dict
  colorectal_cancer_efficientnet.pt   -> dict(model_state_dict, class_names, image_size, ...)
"""
import torch
import torch.nn as nn
from torchvision import models

CLASS_NAMES = ["Adenocarcinoma", "Benign"]      # label 0, label 1 (from training)
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


def load_model(path, device="cpu"):
    """Returns (model, class_names, image_size)."""
    ckpt = _torch_load(path, device)
    meta = {}
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        meta = ckpt
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    num_classes = state["classifier.1.weight"].shape[0]
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(state)
    model.to(device).eval()

    names = meta.get("class_names") or (CLASS_NAMES if num_classes == 2
                                         else [f"class_{i}" for i in range(num_classes)])
    size = int(meta.get("image_size", IMAGE_SIZE))
    return model, list(names), size
