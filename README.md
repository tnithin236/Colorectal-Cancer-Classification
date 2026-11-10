# Colorectal Cancer Classification – Test Project
Binary classifier (Adenocarcinoma vs Benign colon tissue), EfficientNet-B0, trained on LC25000 colon images.

## Setup
1. Copy into `models/`: best_colorectal_model.pt, colorectal_cancer_efficientnet.pt, evaluation_metrics.csv
   (optionally dataset_splits.csv).
2. VS Code terminal:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
3. Ctrl+Shift+P -> Python: Select Interpreter -> choose `.venv`.

## Run
```
streamlit run app.py                                  # dashboard UI: Home / Model Info / Evaluation Metrics / About
python src/inspect_checkpoint.py                      # verify both models load
python src/predict.py --image test_images/x.jpeg
python src/evaluate.py --data_dir test_images         # subfolders colon_aca/ and colon_n/
python src/evaluate.py --splits_csv models/dataset_splits.csv --images_root "D:/path/to/LC25000"
```
Class order (from training): 0 = Adenocarcinoma, 1 = Benign. Preprocessing: resize 224x224 + ImageNet normalization.
