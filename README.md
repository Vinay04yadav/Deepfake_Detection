# Deepfake Video Detector (ResNeXt + LSTM)

## Overview

This project implements and evaluates a deep learning model for detecting deepfake videos. It utilizes a hybrid approach combining a Convolutional Neural Network (CNN - ResNeXt-50) for spatial feature extraction from frames and a Long Short-Term Memory (LSTM) network for analyzing temporal consistency across frames. The goal is to classify a given video as either "Real" or "Fake".

This repository contains the code for a Streamlit-based GUI application for inference and the accompanying Jupyter/Colab notebook detailing the development, training, and evaluation process.

**Note:** This project is shared for educational and informational purposes. Please see the [LICENSE](#license) file for usage terms.

## Features

- Detects deepfakes in video files (MP4, AVI, MOV, MKV).
- Uses a ResNeXt-50 + LSTM deep learning model implemented in PyTorch.
- Provides a classification ('Real' or 'Fake') and a confidence score.
- Simple web interface built with Streamlit for easy video upload and analysis.
- Displays sample frames analyzed by the model.
- Includes a development notebook (`.ipynb`) showing data preparation, training, and evaluation steps.

## Methodology Explained

_(Explain the core concepts here in more detail)_

- **Problem:** The challenge of realistic deepfakes and the need for automated detection.
- **Approach:** Why a hybrid CNN+LSTM model? (CNNs capture spatial details/artifacts within frames, LSTMs capture temporal inconsistencies across frames).
- **Data:** Describe the source dataset (e.g., derived from DFDC via Kaggle), mention it contained pre-cropped faces, and explain _why_ subsetting (to 1200 videos) was necessary (Colab time limits).
- **Preprocessing:** Detail the steps applied (`extract_frames` function): frame extraction, sequence length standardization (150 frames), padding, resizing (112x112), color conversion (BGR->RGB), saving as `.npy`. Explain the purpose of each step.
- **Model Architecture:** Describe the `DeepfakeDetector` class: ResNeXt-50 backbone (transfer learning, feature extraction), LSTM layer (temporal modeling, hidden size), classifier head (pooling, dropout, linear layer).
- **Training:** Explain the process: 80/20 split, `DataLoader`, Cross-Entropy Loss, Adam optimizer, hyperparameters (LR=1e-5, WD=1e-3), 10 epochs, validation-based model saving, use of Colab GPU.
- **Evaluation:** Mention the metrics used (Accuracy, Precision, Recall, F1, AUC, Confusion Matrix) and briefly summarize the results (~83.3% Val Acc, ~90.3% AUC). Refer to the Colab notebook for detailed plots and analysis.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/YourRepositoryName.git](https://github.com/YourUsername/YourRepositoryName.git)
    cd YourRepositoryName
    ```
2.  **Create and activate a Python virtual environment** (Python 3.11 recommended):
    ```bash
    python3.11 -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download the Trained Model:** The trained model weights (`deepfake_detector_best.pth`) are not included in this repository due to size limitations and licensing. You need to download it separately from:
    `[Link to your model file on Google Drive, Dropbox, etc.]`
    Place the downloaded `.pth` file in the main project directory (where `app.py` is located).
    _(Alternatively, if you want users to train it themselves, remove this step and point them to the Colab Notebook)_

## Usage

1.  Ensure your virtual environment is activated and you are in the project directory.
2.  Make sure the `deepfake_detector_best.pth` file is present in the directory.
3.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
4.  The application will open in your web browser.
5.  Use the file uploader to select a video file (MP4, AVI, MOV, MKV).
6.  Click the "Analyze Video" button.
7.  Wait for the analysis to complete. The results (prediction, confidence, metrics, sample frames) will be displayed.

## Colab Notebook

The `Deepfake_Detection_Development.ipynb` notebook contains the full code and explanations for data preprocessing, model definition, training, evaluation, and visualization generation performed using Google Colab.

## Results Summary

The best model achieved the following performance on the validation set (240 videos, 50% real, 50% fake):

- Accuracy: ~83.3%
- AUC: ~90.3%
- F1-Score (Fake Class): ~83.6%
  _(See the Colab notebook for detailed metrics and visualizations like Confusion Matrix, ROC/PR curves etc.)_

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License - see the [LICENSE](LICENSE) file for details. You are free to share and adapt this work for non-commercial purposes, provided you give appropriate credit and license your contributions under the same license.

## Acknowledgements

- Dataset derived from the Deepfake Detection Challenge (DFDC).
- Utilizes PyTorch, OpenCV, Streamlit, Scikit-learn, NumPy, Matplotlib, Seaborn libraries.
- Inspired by initial research on ResNext+LSTM architectures for video analysis.
