---
license: mit
pipeline_tag: video-classification
metrics:
  - accuracy
  - roc_auc
  - precision
  - recall
  - f1
tags:
  - deepfake-detection
  - resnext-50
  - lstm
  - dfdc
  - pytorch
---

# Deepfake Video Detector: ResNeXt-50 + LSTM

## Model summary

A binary video classifier that combines frame-level visual features from a fine-tuned ResNeXt-50 with temporal modeling from a one-layer LSTM. The checkpoint is a PyTorch state dictionary named `deepfake_detector_best.pth`.

**Task:** classify an input clip as `Real` or `Fake`.

## Architecture and inputs

- ResNeXt-50 32×4d initialized with ImageNet-1K V2 pretrained weights; its feature extractor is fine-tuned.
- The CNN produces one 2,048-dimensional feature vector per frame.
- A one-layer LSTM with hidden size 2,048 processes the ordered frame features.
- Temporal average pooling, dropout (0.4), LeakyReLU, and a two-class linear layer produce the Real and Fake logits.
- Input tensor shape: `(batch, 150, 3, 112, 112)`.
- Frames are converted from BGR to RGB, resized to 112×112, and scaled to `[0, 1]`.

The demo reads up to the first 150 decoded frames. For clips shorter than 150 frames, it repeats the final frame to fill the sequence. The training videos contain cropped faces; the demo does not detect or crop faces from arbitrary footage. The displayed softmax scores are relative model scores, not calibrated probabilities.

## Training data

The model was trained on a randomly selected, balanced subset of 1,200 clips (600 Real and 600 Fake) from the [DeepFake Detection Cropped Dataset on Kaggle](https://www.kaggle.com/datasets/ucimachinelearning/deep-fake-detection-cropped-dataset), a DFDC-derived dataset. The notebook uses random seed 42 and a stratified 80/20 split:

| Split | Real | Fake | Total |
| --- | ---: | ---: | ---: |
| Training | 480 | 480 | 960 |
| Validation | 120 | 120 | 240 |

No source videos are included with the checkpoint.

## Training procedure

Training ran for 10 epochs in Google Colab on a Tesla T4 GPU, using cross-entropy loss and Adam with learning rate `1e-5`, weight decay `1e-3`, and batch size 2. The checkpoint with the highest validation accuracy was retained.

## Evaluation

Reported results on the 240-video validation split:

| Metric | Result |
| --- | ---: |
| Accuracy | 83.33% |
| ROC AUC | 90.34% |
| Fake precision | 82.26% |
| Fake recall | 85.00% |
| Fake F1-score | 83.61% |

Confusion matrix (rows are actual labels; columns are predictions):

| Actual | Predicted Real | Predicted Fake |
| --- | ---: | ---: |
| Real | 98 | 22 |
| Fake | 18 | 102 |

The validation split was used to select the checkpoint and report these metrics. Treat them as results for this split, not as an independent estimate of performance on other data.

## Intended use and limitations

This model is intended for research, education, and demonstration. It has been evaluated on one split of one DFDC-derived dataset. Its performance on other datasets, newer manipulation methods, uncropped footage, or real-world videos has not been established. Video compression, lighting, pose, and image quality may affect predictions. Do not use a prediction as proof of authenticity or manipulation.

## Reproduce the demo

The [project repository](https://github.com/Vinay04yadav/Deepfake_Detection) contains the Streamlit app and the model definition. Download `deepfake_detector_best.pth` from this model repository, place it in the project root, install `requirements.txt`, and run:

`streamlit run app.py`

## Paper

“Deepfake Video Detection Using a Combined CNN-LSTM Approach,” 2025 International Conference on Sustainability, Innovation and Technology. [https://doi.org/10.1109/ICSIT65336.2025.11294356](https://doi.org/10.1109/ICSIT65336.2025.11294356)
