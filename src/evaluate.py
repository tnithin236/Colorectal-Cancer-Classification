"""Evaluate the model on labelled images. Two modes:

A) Folder mode (LC25000 layout):
     python src/evaluate.py --data_dir test_images
   test_images/colon_aca/*.jpeg  -> Adenocarcinoma
   test_images/colon_n/*.jpeg    -> Benign
   (folders named Adenocarcinoma / Benign also work)

B) Reproduce the notebook's test split (needs the LC25000 images on this PC):
     python src/evaluate.py --splits_csv models/dataset_splits.csv --images_root "D:/lung_colon_image_set"
   Images are matched by file name, only rows with split == test are used.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, confusion_matrix)
from model_loader import load_model
from preprocess import load_image

EXT = {".jpg", ".jpeg", ".png"}
ALIASES = {"colon_aca": 0, "adenocarcinoma": 0, "aca": 0,
           "colon_n": 1, "benign": 1, "n": 1}


def collect_folder(data_dir):
    paths, labels = [], []
    for d in sorted(p for p in Path(data_dir).iterdir() if p.is_dir()):
        key = d.name.lower()
        if key not in ALIASES:
            print(f"Skipping folder '{d.name}' (expected colon_aca / colon_n / Adenocarcinoma / Benign)")
            continue
        for f in d.rglob("*"):
            if f.suffix.lower() in EXT:
                paths.append(f)
                labels.append(ALIASES[key])
    return paths, labels


def collect_csv(csv, root):
    df = pd.read_csv(csv)
    df = df[df["split"] == "test"]
    index = {f.name: f for f in Path(root).rglob("*") if f.suffix.lower() in EXT}
    paths, labels, missing = [], [], 0
    for p, l in zip(df["image_path"], df["label"]):
        f = index.get(Path(p).name)
        if f is None:
            missing += 1
            continue
        paths.append(f)
        labels.append(int(l))
    if missing:
        print(f"Warning: {missing} test images from the CSV were not found under {root}")
    return paths, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir")
    ap.add_argument("--splits_csv")
    ap.add_argument("--images_root")
    ap.add_argument("--model", default="models/best_colorectal_model.pt")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    if a.splits_csv:
        if not a.images_root:
            raise SystemExit("--images_root is required with --splits_csv")
        paths, y_true = collect_csv(a.splits_csv, a.images_root)
    elif a.data_dir:
        paths, y_true = collect_folder(a.data_dir)
    else:
        raise SystemExit("Give --data_dir or --splits_csv/--images_root")
    if not paths:
        raise SystemExit("No images found.")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, names, size = load_model(a.model, dev)

    probs = []
    with torch.no_grad():
        for i in range(0, len(paths), a.batch):
            x = torch.stack([load_image(p, size) for p in paths[i:i + a.batch]]).to(dev)
            probs.append(torch.softmax(model(x), 1).cpu().numpy())
    probs = np.concatenate(probs)
    y_true = np.array(y_true)
    y_pred = probs.argmax(1)

    print(f"\nImages: {len(paths)}")
    print(f"Accuracy : {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0) * 100:.2f}%")
    print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0) * 100:.2f}%")
    print(f"F1       : {f1_score(y_true, y_pred, zero_division=0) * 100:.2f}%")
    if len(set(y_true)) == 2:
        print(f"ROC-AUC  : {roc_auc_score(y_true, probs[:, 1]):.4f}")
    print("\n" + classification_report(y_true, y_pred, target_names=names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(names)
    ax.set_yticks([0, 1]); ax.set_yticklabels(names)
    for r in range(2):
        for c in range(2):
            ax.text(c, r, cm[r, c], ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    Path("results").mkdir(exist_ok=True)
    plt.savefig("results/confusion_matrix.png", dpi=150)
    print("Saved results/confusion_matrix.png")


if __name__ == "__main__":
    main()
