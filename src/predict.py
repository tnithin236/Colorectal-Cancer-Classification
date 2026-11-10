"""python src/predict.py --image test_images/sample.jpeg   (or a folder)"""
import argparse
from pathlib import Path
import torch
from model_loader import load_model
from preprocess import load_image

EXT = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default="models/best_colorectal_model.pt")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, names, size = load_model(a.model, dev)
    p = Path(a.image)
    files = [f for f in sorted(p.rglob("*")) if f.suffix.lower() in EXT] if p.is_dir() else [p]
    if not files:
        raise SystemExit(f"No images found in {p}")

    with torch.no_grad():
        for f in files:
            x = load_image(f, size).unsqueeze(0).to(dev)
            probs = torch.softmax(model(x), 1)[0]
            print(f"\n{f}")
            for i, n in enumerate(names):
                print(f"  {n:<16} {probs[i].item() * 100:6.2f}%")
            print(f"  -> {names[probs.argmax().item()]}")
    print("\nResearch prototype only - not a clinical diagnosis.")


if __name__ == "__main__":
    main()
