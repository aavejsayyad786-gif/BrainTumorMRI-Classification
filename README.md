# Brain Tumor MRI Image Classification

A student project implementing a custom CNN, MobileNetV2 transfer learning, evaluation, and a Streamlit demo, based on the supplied project guideline.

## Important medical limitation
This is an educational/research demonstration, not a clinically validated diagnostic system. Its outputs must not be used to diagnose, rule out, triage, or make treatment decisions. Model scores are not calibrated clinical probabilities.

## Dataset included
The provided dataset is included under `dataset/` and separated into `train`, `valid`, and `test`. Expected counts are 1,695 training images, 502 validation images, and 246 test images across four labels: `glioma`, `meningioma`, `no_tumor`, and `pituitary`. The test split should be reserved for final evaluation.

## Requirements
- Windows 10/11
- Python version supported by the installed TensorFlow release (Python 3.11 or 3.12 is a conservative option if you encounter package issues)
- Internet connection for downloading MobileNetV2 ImageNet weights on the first transfer-learning run

## Setup (Windows / VS Code)
Open PowerShell in this project directory.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, you can run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and activate again.

## 1. Check the dataset
```powershell
python src/explore_dataset.py
```

## 2. Preview images and data pipeline
```powershell
python src/prepare_data.py
```

## 3. Train the custom CNN
```powershell
python src/train_cnn.py
```
Best model is saved to `models/custom_cnn.keras`; training logs go to `outputs/`.

## 4. Train MobileNetV2 transfer learning model
```powershell
python src/train_transfer_learning.py
```
The first run may download pretrained ImageNet weights. Best model is saved to `models/mobilenetv2.keras`.

## 5. Evaluate and compare
Train one or both models first, then run:
```powershell
python src/evaluate.py
```
Metrics, JSON reports, confusion matrices, and `outputs/model_comparison.csv` are generated for models that exist.

## 6. Run the web application
After training at least one model:
```powershell
python -m streamlit run app.py
```
Choose a model, upload an image, and click **Classify image**.

## Workflow and files
- `src/explore_dataset.py`: counts images per class and split.
- `src/data_utils.py`: shared dataset loader and constants.
- `src/prepare_data.py`: preprocessing check and sample-image preview.
- `src/train_cnn.py`: custom CNN training.
- `src/train_transfer_learning.py`: MobileNetV2 transfer learning and fine-tuning.
- `src/evaluate.py`: test-set metrics, confusion matrices, and model comparison.
- `src/predict.py`: shared prediction helper for Streamlit.
- `app.py`: Streamlit user interface.

## Notes
- Training is not performed automatically during setup; you must run the training scripts to create the model files.
- Use validation data for model development. Do not tune the model repeatedly against the test split.
- Keep a record of the actual measured metrics; do not report placeholder or assumed accuracy.
- Before publishing to GitHub, consider keeping the dataset and generated model files outside the repository if repository size limits are an issue.
