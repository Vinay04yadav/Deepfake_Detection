# ============================================
#         app.py - Deepfake Detector GUI
# ============================================
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
import os
import time
import tempfile # To handle temporary file saving

# ============================================
#         1. Model Definition
# ============================================
# (Copied from Phase 4)
class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2, lstm_hidden_size=2048, num_lstm_layers=1, lstm_dropout=0.4):
        super(DeepfakeDetector, self).__init__()

        # 1. ResNeXt Feature Extractor
        resnext = models.resnext50_32x4d(weights=None) # Load architecture only

        # Remove the final fully connected layer
        modules = list(resnext.children())[:-1]
        self.resnext_features = nn.Sequential(*modules)
        cnn_feature_size = 2048

        # 2. LSTM for Sequence Processing
        self.lstm = nn.LSTM(
            input_size=cnn_feature_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if num_lstm_layers > 1 else 0
        )

        # 3. Classifier Head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(lstm_dropout)
        self.relu = nn.LeakyReLU()
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        # x shape: (B, N, C, H, W)
        batch_size, seq_length, C, H, W = x.shape
        x_reshaped = x.view(batch_size * seq_length, C, H, W)
        features = self.resnext_features(x_reshaped) # (B*N, 2048, 1, 1)
        features = features.view(batch_size * seq_length, -1) # (B*N, 2048)
        features_seq = features.view(batch_size, seq_length, -1) # (B, N, 2048)
        lstm_out, _ = self.lstm(features_seq) # (B, N, 2048)
        lstm_out_permuted = lstm_out.permute(0, 2, 1) # (B, 2048, N)
        pooled_out = self.pool(lstm_out_permuted) # (B, 2048, 1)
        pooled_out = pooled_out.squeeze(-1) # (B, 2048)
        out = self.dropout(pooled_out)
        out = self.relu(out)
        logits = self.fc(out) # (B, 2)
        return logits

# ============================================
#         2. Helper Functions
# ============================================
# (Copied from Phase 2)
def extract_frames(video_path, frame_count, img_size):
    """
    Extracts, resizes, and pads frames from a video file.
    Args are video_path, target frame_count, target img_size (height, width).
    Returns np.ndarray (frame_count, H, W, 3) or None.
    """
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None

        total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames_in_video == 0:
             print(f"Warning: Video {video_path} has 0 frames.")
             cap.release()
             return None

        extracted_count = 0
        while extracted_count < frame_count:
            ret, frame = cap.read()
            if not ret: break # Video ended or error

            # Resize frame
            frame_resized = cv2.resize(frame, (img_size[1], img_size[0])) # cv2 uses (width, height)
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            extracted_count += 1

        cap.release()

        if extracted_count == 0:
             print(f"Warning: Could not read any frames from {video_path}.")
             return None

        # Pad if necessary
        if extracted_count < frame_count:
            last_frame = frames[-1]
            padding_needed = frame_count - extracted_count
            for _ in range(padding_needed):
                frames.append(last_frame)

        # Ensure output is always float32 for consistency before tensor conversion
        return np.array(frames, dtype=np.float32) # Shape: (frame_count, H, W, 3)

    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        return None

# --- Model Loading Function ---
@st.cache_resource # Cache the loaded model (important for performance)
def load_model(model_path, device):
    print(f"Attempting to load model from {model_path} onto {device}...")
    model = DeepfakeDetector()
    try:
      # Load weights, ensuring compatibility between CPU/GPU saves/loads
      model.load_state_dict(torch.load(model_path, map_location=device))
      model.to(device)
      model.eval() # Set to evaluation mode
      print("Model loaded successfully.")
      return model
    except Exception as e:
      print(f"Error loading model weights from {model_path}: {e}")
      # Raise the error to be caught in the main app part
      raise e

# --- Prediction Function (Modified to return sample frames) ---
def predict_video(video_path, model, device, frame_count, img_size):
    """
    Predicts if a video is Real or Fake.
    Returns: (label_str, confidence_float, sample_frames_list or None)
    """
    # 1. Preprocessing
    sample_frames_to_return = 4 # How many frames to show
    frame_sequence = extract_frames(video_path, frame_count, img_size)
    if frame_sequence is None:
        return "Error: Could not extract frames.", 0.0, None

    # Select sample frames BEFORE converting to tensor (easier for display)
    # Ensure sequence has enough frames for indices
    num_actual_frames = frame_sequence.shape[0]
    # Ensure indices are within bounds
    indices = np.linspace(0, num_actual_frames - 1, sample_frames_to_return, dtype=int)
    # Ensure frames are uint8 for display if they aren't already
    sample_frames = [(frame_sequence[i] ).astype(np.uint8) for i in indices] # List of NumPy arrays (H, W, C), ensure uint8

    try:
        # Prepare tensor for model (scale to [0, 1])
        sequence_tensor = torch.from_numpy(frame_sequence / 255.0) # Already float32 from extract_frames
        sequence_tensor = sequence_tensor.permute(0, 3, 1, 2) # (N, C, H, W)
        sequence_tensor = sequence_tensor.unsqueeze(0) # Add batch dim: (1, N, C, H, W)
        sequence_tensor = sequence_tensor.to(device)
    except Exception as e:
        return f"Error during tensor conversion: {e}", 0.0, None

    # 2. Inference
    try:
        with torch.no_grad():
            outputs = model(sequence_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)
    except Exception as e:
        return f"Error during model inference: {e}", 0.0, None

    # 3. Post-processing
    predicted_idx = predicted_idx.item()
    confidence_score = confidence.item()
    label_map = {0: 'Real', 1: 'Fake'}
    predicted_label_str = label_map.get(predicted_idx, "Unknown")

    return predicted_label_str, confidence_score, sample_frames


# ============================================
#         3. Streamlit App Interface
# ============================================

# --- Page Configuration ---
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration ---
# !! IMPORTANT: Make sure this path is correct relative to where you run app.py !!
MODEL_PATH = 'deepfake_detector_best.pth'
FRAME_COUNT = 150             # Should match training
IMG_SIZE = (112, 112)         # Should match training (Height, Width)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Load Model ---
# Placed here to load early and show potential errors clearly
try:
    model = load_model(MODEL_PATH, device)
    # Optionally display model load success in sidebar after it loads
    st.sidebar.success(f"Model loaded on {device}.")
except FileNotFoundError:
    st.error(f"Fatal Error: Model file not found at '{MODEL_PATH}'.")
    st.error("Please ensure the '.pth' file is in the same directory as 'app.py'.")
    st.stop() # Stop execution if model isn't found
except Exception as e:
    st.error(f"Fatal Error loading the model: {e}")
    st.stop()


# --- Sidebar Content ---
st.sidebar.header("About")
st.sidebar.info(
    "This application uses a Deep Learning model (ResNeXt50 + LSTM) "
    "to detect whether a video is likely real or a deepfake. "
    "Upload a video file and click 'Analyze Video' to get a prediction."
)
st.sidebar.markdown("---")
st.sidebar.header("Key Features (Model)")
st.sidebar.markdown("""
- **Architecture:** ResNeXt50 (for frame features) + LSTM (for temporal analysis)
- **Input:** Analyzes the first 150 frames of size 112x112.
- **Output:** Classifies as 'Real' or 'Fake' with a confidence score.
""")
st.sidebar.markdown("---")


# --- Main Page Content ---

# Centered Title using Markdown
st.markdown("<h1 style='text-align: center; color: #00A0A0;'>Deepfake Video Detector</h1>", unsafe_allow_html=True)
st.write(
    "Welcome! Protect yourself from digital deception. "
    "Upload your video for AI-powered analysis to detect potential manipulation."
)
st.divider() # Visual separator

# --- File Upload Section ---
uploaded_file = st.file_uploader(
    "Choose a video file for analysis:",
    type=["mp4", "avi", "mov", "mkv"], # Allowed types
    help="Upload a video file (MP4, AVI, MOV, MKV)."
)

st.divider()

# --- Analysis Trigger and Results ---
if uploaded_file is not None:
    # Display filename and optional preview
    st.info(f"Uploaded File: **{uploaded_file.name}**")
    # Display video only after upload, before analysis button
    st.video(uploaded_file)

    # Analyze Button
    if st.button("Analyze Video", type="primary"): # Make button more prominent
        # Save uploaded file temporarily for OpenCV access
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_video_path = tmp_file.name

        # --- Prediction ---
        # Show spinner during processing
        with st.spinner('🧠 Analyzing video... This may take a few moments...'):
            start_time = time.time()
            # --- Call the prediction function ---
            predicted_label, confidence, sample_frames = predict_video(
                temp_video_path, model, device, FRAME_COUNT, IMG_SIZE
            )
            end_time = time.time()
            processing_time = end_time - start_time

        # --- Display Results Section ---
        st.subheader("📊 Analysis Results")
        if "Error" in predicted_label:
             st.error(predicted_label)
        else:
            # --- Display Metrics ---
            st.success(f"Analysis complete in {processing_time:.2f} seconds.")
            col1, col2, col3 = st.columns(3)
            col1.metric("Frames Analyzed", f"{FRAME_COUNT}")
            col2.metric("Processing Time", f"{processing_time:.2f}s")
            col3.metric("Model Confidence", f"{confidence:.2%}") # Format as percentage

            # Display Overall Prediction using status boxes
            st.markdown("#### Overall Prediction:")
            if predicted_label == "Fake":
                st.error(f"🚨 Result: Predicted as **{predicted_label}**")
            else: # Real or Unknown
                st.success(f"✅ Result: Predicted as **{predicted_label}**")
            # Show confidence as a progress bar
            st.progress(confidence)
            st.caption(f"Confidence score: {confidence:.4f}")


            st.divider() # Visual separator

            # --- Display Sample Frames ---
            if sample_frames:
                st.subheader("🖼️ Sample Analyzed Frames")
                st.write("_Showing a few frames processed by the model._")
                # Create columns - number matches frames returned
                frame_cols = st.columns(len(sample_frames))
                for i, frame in enumerate(sample_frames):
                    with frame_cols[i]:
                        # Display frame, ensure color channels are correct if needed
                        # Assuming frame is HWC, RGB uint8
                        st.image(
                            frame,
                            caption=f'Frame approx. #{int(i * FRAME_COUNT / len(sample_frames))}',
                            use_container_width=True # <<< USE THIS NEW PARAMETER INSTEAD
                        )
            else:
                st.warning("Could not retrieve sample frames for display.")

        # --- Clean up temporary file ---
        try:
            os.remove(temp_video_path)
        except Exception as e:
            st.warning(f"Could not remove temporary file {temp_video_path}: {e}")
else:
    # Message when no file is uploaded yet
    st.info("☝️ Upload a video file above to begin analysis.")


# --- Footer / Optional Section ---
st.divider()
st.markdown("---")
st.caption("Deepfake Detection Project | Model: ResNeXt50+LSTM")