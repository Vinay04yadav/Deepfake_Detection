---
pipeline_tag: video-classification
library_name: pytorch
tags:
  - deepfake-detection
  - video-classification
  - resnext
  - lstm
---

# Deepfake Video Detector (ResNeXt-50 + LSTM)

## Status

This is a research prototype associated with the paper “Deepfake Video Detection Using a Combined CNN-LSTM Approach.” The model checkpoint is not currently published. Its license and redistribution rights need to be confirmed before the weights are released.

## Model details

- **Task:** Binary video classification (Real or Fake)
- **Architecture:** ResNeXt-50 frame feature extractor followed by an LSTM sequence classifier
- **Framework:** PyTorch
- **Input:** A sequence of up to 150 frames, resized to 112 × 112 pixels
- **Output:** Scores for the Real and Fake classes

The current application samples frames from the beginning of the uploaded clip. It expects face-cropped input like the training data and does not detect or crop faces in arbitrary footage.

## Training data

The notebook uses a 1,200-video subset (600 Real and 600 Fake) of the Kaggle DeepFake Detection Cropped Dataset, which is derived from DFDC. The subset was split into 960 training videos and 240 validation videos (120 per class).

The dataset is subject to its own terms. This model card does not grant rights to the source videos or establish that the trained checkpoint may be redistributed. No dataset videos are included in this repository.

## Evaluation

Reported results on the notebook's stratified 240-video validation split:

| Metric | Result |
| --- | ---: |
| Accuracy | 83.33% |
| ROC AUC | 90.34% |
| Fake precision | 82.26% |
| Fake recall | 85.00% |
| Fake F1-score | 83.61% |

This is one validation split from one dataset source. The same validation split was used to select the best checkpoint and report these metrics, so these figures are not an independent test estimate.

## Intended use and limitations

Use this model for research, education, and demonstration of the documented CNN-LSTM workflow. Do not treat its predictions as proof that a video is authentic or manipulated, or use them as the sole basis for consequential decisions.

The model was trained on a subset derived from DFDC. It is not established to work on recent deepfake generation methods, other datasets, un-cropped footage, or real-world videos. Distribution shifts, compression, lighting, pose, and video quality can affect its output. A “Real” prediction does not certify authenticity, and a “Fake” prediction does not prove manipulation.

## License

No license is granted for the model weights at this time. The source code license, dataset terms, and paper rights are separate. Add a Hugging Face `license` metadata field only after confirming the rights that apply to the checkpoint.
