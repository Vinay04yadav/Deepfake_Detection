from pathlib import Path
import os
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models


st.set_page_config(
    page_title="Deepfake Video Detection",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "deepfake_detector_best.pth"
PAPER_URL = "https://doi.org/10.1109/ICSIT65336.2025.11294356"
FRAME_COUNT = 150
IMG_SIZE = (112, 112)
PAGES = ["Home", "Methodology", "Results", "Try yourself"]


class DeepfakeDetector(nn.Module):
    """ResNeXt frame features followed by an LSTM video classifier."""

    def __init__(self, num_classes=2, lstm_hidden_size=2048, num_lstm_layers=1, lstm_dropout=0.4):
        super().__init__()
        resnext = models.resnext50_32x4d(weights=None)
        self.resnext_features = nn.Sequential(*list(resnext.children())[:-1])
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if num_lstm_layers > 1 else 0,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(lstm_dropout)
        self.relu = nn.LeakyReLU()
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        batch_size, sequence_length, channels, height, width = x.shape
        frames = x.reshape(batch_size * sequence_length, channels, height, width)
        features = self.resnext_features(frames).reshape(batch_size, sequence_length, -1)
        sequence, _ = self.lstm(features)
        pooled = self.pool(sequence.permute(0, 2, 1)).squeeze(-1)
        return self.fc(self.relu(self.dropout(pooled)))


def extract_frames(video_path, frame_count=FRAME_COUNT, img_size=IMG_SIZE):
    """Read the first frames, resize to the training shape, and pad short clips."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None

    frames = []
    try:
        while len(frames) < frame_count:
            ok, frame = capture.read()
            if not ok:
                break
            resized = cv2.resize(frame, (img_size[1], img_size[0]))
            frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    if not frames:
        return None
    if len(frames) < frame_count:
        frames.extend([frames[-1]] * (frame_count - len(frames)))
    return np.asarray(frames, dtype=np.float32)


@st.cache_resource(show_spinner=False)
def load_model(model_path, device_name):
    model = DeepfakeDetector()
    state = torch.load(model_path, map_location=device_name)
    model.load_state_dict(state)
    model.to(device_name)
    model.eval()
    return model


def predict_video(video_path, model, device):
    frames = extract_frames(video_path)
    if frames is None:
        raise ValueError("This file could not be decoded as a video. Try another MP4, AVI, MOV, or MKV file.")

    display_frames = [frames[i].astype(np.uint8) for i in np.linspace(0, FRAME_COUNT - 1, 4, dtype=int)]
    sequence = torch.from_numpy(frames / 255.0).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    with torch.inference_mode():
        scores = torch.softmax(model(sequence), dim=1)[0].cpu().numpy()
    label = "Fake" if int(np.argmax(scores)) == 1 else "Real"
    return label, float(scores[0]), float(scores[1]), display_frames


st.markdown(
    """
    <style>
      :root { --ink:#162f36; --muted:#536b70; --paper:#f3f5f1; --line:#d6e0dc;
              --teal:#087f78; --teal-dark:#075e5a; --mist:#e4efeb; --sand:#f4e5d8;
              --coral:#c97853; --white:#fffefa; }
      .stApp { background:var(--paper); color:var(--ink); }
      [data-testid="stHeader"] { background:rgba(243,245,241,.94); }
      [data-testid="stSidebar"] { display:none; }
      .block-container { max-width:1220px; padding:1.25rem 2rem 4rem; }
      h1,h2,h3 { color:var(--ink); font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; letter-spacing:-.03em; }
      h1 { font-weight:600; }
      p,li { color:#405459; }
      .brand { font-size:1rem; font-weight:650; color:var(--ink); }
      .brand small { display:block; margin-top:.1rem; color:var(--muted); font-size:.72rem; font-weight:500; letter-spacing:.02em; }
      [data-testid="stButton"] button { min-height:2.75rem; border-radius:8px; font-size:.9rem; transition:transform .16s ease, box-shadow .16s ease, background-color .16s ease; }
      [data-testid="stButton"] button:hover { transform:translateY(-1px); box-shadow:0 5px 14px rgba(22,47,54,.12); }
      [data-testid="stBaseButton-primary"], button[kind="primary"] { background:var(--teal)!important; border-color:var(--teal)!important; color:#fff!important; }
      [data-testid="stBaseButton-primary"] p, button[kind="primary"] p { color:#fff!important; }
      [data-testid="stBaseButton-secondary"], button[kind="secondary"] { background:var(--white)!important; border-color:#bfd0ca!important; color:var(--ink)!important; }
      [data-testid="stBaseButton-secondary"] p, button[kind="secondary"] p { color:var(--ink)!important; }
      [data-testid="stLinkButton"] a { border-radius:8px; color:var(--teal-dark)!important; border-color:#bfd0ca!important; background:var(--white)!important; }
      .hero { margin:1.25rem 0 1.1rem; padding:clamp(2rem,5vw,4rem); border:1px solid #d2dfda;
              border-radius:18px; background:linear-gradient(118deg,#e3eeea 0%,#f7f5ef 70%); }
      .hero-grid { display:grid; grid-template-columns:1.08fr .92fr; gap:3rem; align-items:center; }
      .hero h1 { font-size:clamp(2.65rem,5.4vw,4.8rem); line-height:1.02; margin:.8rem 0 1rem; max-width:12ch; text-wrap:balance; }
      .hero p { font-size:1.08rem; line-height:1.65; max-width:58ch; margin:0; }
      .diagram { padding:1.35rem; border:1px solid #cadbd5; border-radius:12px; background:rgba(255,254,250,.82); box-shadow:0 14px 32px rgba(22,47,54,.06); }
      .diagram-title { color:var(--muted); font-size:.84rem; font-weight:550; margin-bottom:1rem; }
      .flow-row { display:flex; align-items:center; gap:.5rem; margin:.65rem 0; }
      .flow-node { flex:1; min-height:3.1rem; display:flex; align-items:center; justify-content:center; text-align:center;
                   padding:.5rem; background:#eaf2ef; border:1px solid #d4e2dc; border-radius:8px; color:var(--ink); font-size:.86rem; }
      .flow-arrow { color:var(--teal); font-weight:700; }
      .section-title { font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; font-weight:600; font-size:clamp(2rem,3.8vw,3.1rem); line-height:1.08; letter-spacing:-.03em; margin:.45rem 0 .7rem; text-wrap:balance; }
      .section-copy { color:#52666a; font-size:1.02rem; line-height:1.65; max-width:68ch; }
      .stat { padding:1rem 0; border-top:2px solid var(--teal); }
      .stat strong { display:block; color:var(--ink); font-size:2rem; font-weight:600; font-variant-numeric:tabular-nums; }
      .stat span { color:var(--muted); font-size:.86rem; }
      .note { padding:1rem 1.1rem; border-left:3px solid var(--teal); background:#e6efeb; border-radius:0 8px 8px 0; color:#304a50; line-height:1.6; }
      .pipeline-step { height:100%; min-height:9rem; padding:1.1rem; background:var(--white); border:1px solid var(--line); border-radius:10px; }
      .pipeline-step b { display:block; margin:.5rem 0; color:var(--ink); font-size:1rem; }
      .pipeline-step small { display:block; color:var(--muted); line-height:1.55; font-size:.86rem; }
      .step-number { color:var(--teal-dark); font-size:.82rem; font-weight:650; }
      .cta-gap { height:1rem; }
      .section-rule { height:1px; margin:2.25rem 0; background:var(--line); }
      .data-tile { min-height:8rem; padding:1.2rem; background:var(--white); border:1px solid var(--line); border-radius:10px; }
      .data-tile strong { display:block; margin:.25rem 0; color:var(--ink); font-size:1.7rem; font-variant-numeric:tabular-nums; }
      .data-tile span { color:var(--muted); font-size:.88rem; line-height:1.5; }
      .architecture { margin:1.5rem 0; padding:1.4rem; border:1px solid var(--line); border-radius:14px; background:var(--white); }
      .architecture-title { margin-bottom:1rem; color:var(--muted); font-size:.9rem; }
      .architecture-flow { display:grid; grid-template-columns:1fr auto 1fr auto 1fr auto 1fr; gap:.65rem; align-items:stretch; }
      .architecture-node { padding:1rem .85rem; border-radius:9px; background:#eaf2ef; border:1px solid #d6e4de; }
      .architecture-node.accent { background:#e4eeeb; border-color:#bdd4cb; }
      .architecture-node b { display:block; margin:.4rem 0; color:var(--ink); }
      .architecture-node small { display:block; color:var(--muted); line-height:1.5; }
      .architecture-arrow { display:flex; align-items:center; color:var(--coral); font-size:1.35rem; }
      .filmstrip { display:flex; gap:.45rem; margin:.8rem 0 1.2rem; }
      .film-frame { position:relative; flex:1; height:4.1rem; overflow:hidden; border:1px solid #cadbd5; border-radius:7px; background:linear-gradient(145deg,#dce9e4,#f1e9dc); }
      .film-frame:before { content:""; position:absolute; width:1.55rem; height:1.8rem; left:50%; top:.55rem; transform:translateX(-50%); border-radius:48% 48% 43% 43%; background:#66847d; }
      .film-frame:after { content:""; position:absolute; width:2.4rem; height:1.15rem; left:50%; bottom:-.45rem; transform:translateX(-50%); border-radius:50% 50% 0 0; background:#91aaa1; }
      .matrix { display:grid; grid-template-columns:1.05fr 1fr 1fr; gap:.45rem; margin:1rem 0; }
      .matrix-label { display:flex; align-items:center; justify-content:center; min-height:2.8rem; padding:.5rem; text-align:center; color:var(--muted); font-size:.82rem; }
      .matrix-cell { min-height:6rem; display:flex; flex-direction:column; justify-content:center; padding:.8rem; border-radius:9px; text-align:center; }
      .matrix-cell strong { color:var(--ink); font-size:1.8rem; font-variant-numeric:tabular-nums; }
      .matrix-cell span { color:#354e53; font-size:.8rem; }
      .matrix-hit { background:linear-gradient(145deg,#d6e9e1,#e6f0eb); border:1px solid #c3dcd1; }
      .matrix-error { background:linear-gradient(145deg,#f4e6d9,#f8eee6); border:1px solid #ead7c6; }
      .result-band { padding:1.1rem 1.3rem; border:1px solid #d6e2dc; border-radius:10px; background:#e8f0ec; line-height:1.6; }
      .demo-grid { display:grid; grid-template-columns:1fr 1fr; gap:1rem; align-items:stretch; }
      .demo-visual { position:relative; min-height:21rem; overflow:hidden; padding:1.5rem; border:1px solid #c9d9d3; border-radius:14px; background:radial-gradient(ellipse at 80% 15%,rgba(201,120,83,.24),transparent 35%),radial-gradient(ellipse at 20% 90%,rgba(8,127,120,.2),transparent 38%),#e7efeb; background-size:140% 140%; animation:drift 15s ease-in-out infinite alternate; }
      @keyframes drift { from { background-position:0% 20%; } to { background-position:100% 80%; } }
      .demo-visual h3 { margin:.35rem 0 .7rem; font-size:1.5rem; }
      .demo-visual p { max-width:40ch; color:#425b5e; line-height:1.6; }
      .demo-visual .flow-node { background:rgba(255,254,250,.72); }
      .demo-meta { display:grid; grid-template-columns:1fr 1fr; gap:.6rem; margin-top:1.2rem; }
      .demo-meta div { padding:.8rem; border:1px solid rgba(130,159,149,.35); border-radius:8px; background:rgba(255,254,250,.55); }
      .demo-meta b { display:block; color:var(--ink); font-size:1.05rem; }
      .demo-meta span { color:var(--muted); font-size:.78rem; }
      .footer { margin-top:3rem; padding-top:1rem; border-top:1px solid var(--line); color:var(--muted); font-size:.8rem; }
      @media (prefers-reduced-motion: reduce) { .demo-visual { animation:none; } [data-testid="stButton"] button { transition:none; } }
      @media (max-width:760px) {
        .block-container { padding-left:1rem; padding-right:1rem; }
        .hero-grid { grid-template-columns:1fr; gap:1.6rem; }
        .hero { padding:1.5rem; }
        .architecture-flow { grid-template-columns:1fr; }
        .architecture-arrow { justify-content:center; transform:rotate(90deg); }
        .demo-grid { grid-template-columns:1fr; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def go_to(page):
    st.session_state["navigation"] = page


def section_heading(title, copy=None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if copy:
        st.markdown(f'<div class="section-copy">{copy}</div>', unsafe_allow_html=True)


header_brand, header_nav, header_paper = st.columns([1.4, 4.2, 1.35])
with header_brand:
    st.markdown('<div class="brand">Deepfake video detector<small>ResNeXt-50 + LSTM research prototype</small></div>', unsafe_allow_html=True)
with header_nav:
    if "navigation" not in st.session_state:
        st.session_state.navigation = "Home"
    nav_columns = st.columns(len(PAGES))
    for nav_column, nav_page in zip(nav_columns, PAGES):
        with nav_column:
            st.button(
                nav_page,
                type="primary" if st.session_state.navigation == nav_page else "secondary",
                use_container_width=True,
                on_click=go_to,
                args=(nav_page,),
                key=f"nav_{nav_page.lower().replace(' ', '_')}",
            )
with header_paper:
    st.link_button("Read the paper ↗", PAPER_URL, use_container_width=True)

page = st.session_state.navigation

if page == "Home":
    st.markdown(
        """
        <section class="hero"><div class="hero-grid">
          <div>
            <h1>Detecting deepfakes across frames</h1>
            <p>A video deepfake detector that pairs ResNeXt-50 frame features with LSTM sequence modeling. Explore the method, review its validation results, or try the model on a clip.</p>
          </div>
          <div class="diagram">
            <div class="diagram-title">A video contains spatial and temporal signals</div>
            <div class="flow-row"><div class="flow-node">150 video frames</div><div class="flow-arrow">→</div><div class="flow-node">ResNeXt-50<br>frame features</div></div>
            <div class="flow-row"><div class="flow-node">2048 features<br>per frame</div><div class="flow-arrow">→</div><div class="flow-node">LSTM<br>sequence model</div></div>
            <div class="flow-row"><div class="flow-node">Video-level<br>classification</div><div class="flow-arrow">→</div><div class="flow-node">Real / Fake</div></div>
          </div>
        </div></section>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown('<div class="stat"><strong>83.33%</strong><span>Validation accuracy on 240 videos</span></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="stat"><strong>90.34%</strong><span>validation ROC AUC</span></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="stat"><strong>1,200</strong><span>balanced videos in the study subset</span></div>', unsafe_allow_html=True)

    st.write("")
    section_heading("From research to a working demo", "The project combines a documented training workflow with an interactive video analysis interface.")
    st.write("")
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="pipeline-step"><span>Step 1</span><b>How the detector works</b><small>See the data preparation steps and how frame-level features become a video prediction.</small></div>', unsafe_allow_html=True)
        st.markdown('<div class="cta-gap"></div>', unsafe_allow_html=True)
        st.button("Read the methodology", on_click=go_to, args=("Methodology",), key="home_method")
    with b:
        st.markdown('<div class="pipeline-step"><span>Step 2</span><b>What the evaluation shows</b><small>Review validation metrics, class-wise performance, and the limits of this experiment.</small></div>', unsafe_allow_html=True)
        st.markdown('<div class="cta-gap"></div>', unsafe_allow_html=True)
        st.button("View the results", on_click=go_to, args=("Results",), key="home_results")
    with c:
        st.markdown('<div class="pipeline-step"><span>Step 3</span><b>Analyze a video clip</b><small>Upload a supported video and inspect the model output and example frames.</small></div>', unsafe_allow_html=True)
        st.markdown('<div class="cta-gap"></div>', unsafe_allow_html=True)
        st.button("Try the detector", on_click=go_to, args=("Try yourself",), type="primary", key="home_try")

elif page == "Methodology":
    section_heading("How each video becomes a prediction", "The pipeline keeps the order of a clip intact: a CNN describes each frame, then an LSTM reads how those descriptions change over time.")
    st.write("")
    d1, d2, d3 = st.columns(3)
    for col, value, label in [
        (d1, "3,293", "videos in the downloaded cropped-face dataset"),
        (d2, "1,200", "balanced videos sampled for this experiment"),
        (d3, "960 / 240", "training and validation videos"),
    ]:
        with col:
            st.markdown(f'<div class="data-tile"><span>{label}</span><strong>{value}</strong></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="architecture">
      <div class="architecture-title">Spatial features flow into temporal classification</div>
      <div class="filmstrip" aria-label="A sequence of video frames">
        <div class="film-frame"></div><div class="film-frame"></div><div class="film-frame"></div>
        <div class="film-frame"></div><div class="film-frame"></div><div class="film-frame"></div>
      </div>
      <div class="architecture-flow">
        <div class="architecture-node"><span class="step-number">Input</span><b>150 RGB frames</b><small>112 × 112 pixels. The first frames are used; short clips repeat the last frame.</small></div>
        <div class="architecture-arrow">→</div>
        <div class="architecture-node accent"><span class="step-number">Spatial encoder</span><b>ResNeXt-50</b><small>Fine-tuned per frame. Produces one 2,048-dimensional feature vector for each frame.</small></div>
        <div class="architecture-arrow">→</div>
        <div class="architecture-node accent"><span class="step-number">Temporal model</span><b>One-layer LSTM</b><small>Reads the ordered feature sequence with a hidden size of 2,048.</small></div>
        <div class="architecture-arrow">→</div>
        <div class="architecture-node"><span class="step-number">Video output</span><b>Real or Fake</b><small>Temporal average pooling, dropout, LeakyReLU, and a two-class linear head.</small></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Data preparation")
    prep1, prep2, prep3 = st.columns(3)
    prep_steps = [
        (prep1, "Frame extraction", "OpenCV reads consecutive frames from the beginning of each face-cropped clip. A clip with fewer than 150 frames is padded by repeating its last frame."),
        (prep2, "Image conversion", "Every frame is resized to 112 × 112 pixels and converted from OpenCV’s BGR order to RGB."),
        (prep3, "Balanced split", "The notebook samples 600 Real and 600 Fake videos, then uses a stratified split: 480 per class for training and 120 per class for validation."),
    ]
    for col, title, description in prep_steps:
        with col:
            st.markdown(f'<div class="pipeline-step"><span class="step-number">{title}</span><small>{description}</small></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.subheader("Model training and evaluation")
    train_col, eval_col = st.columns([1.05, .95])
    with train_col:
        st.markdown("""
        The notebook fine-tunes an ImageNet-pretrained ResNeXt-50 with a one-layer LSTM. Training uses **cross-entropy loss**, the **Adam optimizer**, a learning rate of **1e-5**, and weight decay of **1e-3**. The best checkpoint is selected by validation accuracy over 10 epochs.

        The class head averages the LSTM outputs across time, then applies dropout, LeakyReLU, and a linear layer for the Real and Fake scores.
        """)
    with eval_col:
        st.markdown("""
        <div class="note"><b>What this evaluation covers</b><br><br>
        Results come from one stratified 80/20 split of 1,200 videos. The validation set contains 120 Real and 120 Fake clips. The demo analyzes the first 150 frames and does not detect or crop faces in arbitrary footage.<br><br>
        Treat the score as a research result on this dataset, not as a guarantee for other sources or manipulations.
        </div>
        """, unsafe_allow_html=True)

elif page == "Results":
    section_heading("Validation performance", "Reported results for the best checkpoint on a stratified validation split of 240 videos: 120 Real and 120 Fake.")
    st.write("")
    r1, r2, r3, r4, r5 = st.columns(5)
    for col, label, value in [
        (r1, "Accuracy", "83.33%"),
        (r2, "ROC AUC", "90.34%"),
        (r3, "Fake precision", "82.26%"),
        (r4, "Fake recall", "85.00%"),
        (r5, "Fake F1-score", "83.61%"),
    ]:
        with col:
            st.markdown(f'<div class="stat"><strong>{value}</strong><span>{label}</span></div>', unsafe_allow_html=True)

    st.write("")
    matrix_col, detail_col = st.columns([.9, 1.1])
    with matrix_col:
        st.subheader("Validation confusion matrix")
        st.caption("Rows show the known label. Columns show the model prediction.")
        st.markdown("""
        <div class="matrix" role="img" aria-label="Confusion matrix. Actual Real: 101 predicted Real and 19 predicted Fake. Actual Fake: 17 predicted Real and 103 predicted Fake.">
          <div class="matrix-label"></div><div class="matrix-label">Predicted Real</div><div class="matrix-label">Predicted Fake</div>
          <div class="matrix-label">Actual Real</div><div class="matrix-cell matrix-hit"><strong>101</strong><span>correct Real</span></div><div class="matrix-cell matrix-error"><strong>19</strong><span>Real marked Fake</span></div>
          <div class="matrix-label">Actual Fake</div><div class="matrix-cell matrix-error"><strong>17</strong><span>Fake marked Real</span></div><div class="matrix-cell matrix-hit"><strong>103</strong><span>correct Fake</span></div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("120 Real clips and 120 Fake clips in the validation split.")
    with detail_col:
        st.subheader("Reading these numbers")
        st.markdown("""
        - **204 of 240 videos** were classified correctly.
        - **103 of 120 fake videos** were detected as Fake.
        - **19 of 120 real videos** were incorrectly classified as Fake.
        - Fake-class F1-score: **83.61%**.
        """)
        st.markdown('<div class="note">These are validation-set results from one split of a balanced subset. They do not establish performance on every deepfake source, editing method, or real-world video.</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="result-band"><b>Publication</b><br>“Deepfake Video Detection Using a Combined CNN-LSTM Approach”<br>2025 International Conference on Sustainability, Innovation and Technology</div>', unsafe_allow_html=True)
    st.link_button("Read the paper on IEEE Xplore", PAPER_URL)

elif page == "Try yourself":
    section_heading("Upload a video clip", "The app analyzes up to 150 frames from the beginning of the video. Processing time depends on the clip and available hardware.")
    st.write("")
    if not MODEL_PATH.is_file():
        st.error(f"Model checkpoint not found at {MODEL_PATH.name}. Add the checkpoint next to app.py to enable inference.")
    upload_col, visual_col = st.columns([1.05, .95], gap="large")
    with upload_col:
        st.markdown("#### Select a clip")
        uploaded_file = st.file_uploader(
            "Choose a video",
            type=["mp4", "avi", "mov", "mkv"],
            help="Supported formats: MP4, AVI, MOV, and MKV.",
            disabled=not MODEL_PATH.is_file(),
            key="video_upload",
        )
        if uploaded_file is not None:
            if st.session_state.get("analysis_filename") != uploaded_file.name:
                st.session_state.analysis_result = None
                st.session_state.analysis_error = None
            st.video(uploaded_file.getvalue())
            st.caption(f"{uploaded_file.name}: {uploaded_file.size / (1024 * 1024):.1f} MB")

            if st.button("Analyze video", type="primary", disabled=not MODEL_PATH.is_file(), use_container_width=True):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_path = temp_file.name

                    with st.spinner("Loading the detector and analyzing the video…"):
                        started = time.perf_counter()
                        model = load_model(str(MODEL_PATH), str(device))
                        label, real_score, fake_score, sample_frames = predict_video(temp_path, model, device)
                        elapsed = time.perf_counter() - started
                    st.session_state.analysis_filename = uploaded_file.name
                    st.session_state.analysis_result = {
                        "label": label,
                        "real_score": real_score,
                        "fake_score": fake_score,
                        "sample_frames": sample_frames,
                        "elapsed": elapsed,
                    }
                    st.session_state.analysis_error = None
                except Exception as error:
                    st.session_state.analysis_result = None
                    st.session_state.analysis_error = str(error)
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)

    with visual_col:
        st.markdown("""
        <div class="demo-visual">
          <span class="step-number">Inside the analysis window</span>
          <h3>One short segment, read in sequence</h3>
          <p>The model compares frame-level visual features across time. It does not scan every frame in a long video.</p>
          <div class="filmstrip"><div class="film-frame"></div><div class="film-frame"></div><div class="film-frame"></div><div class="film-frame"></div><div class="film-frame"></div></div>
          <div class="flow-row"><div class="flow-node">Frame features</div><div class="flow-arrow">→</div><div class="flow-node">LSTM sequence</div><div class="flow-arrow">→</div><div class="flow-node">Two scores</div></div>
          <div class="demo-meta">
            <div><b>150</b><span>frames from the clip start</span></div>
            <div><b>112 × 112</b><span>pixels per frame</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="note" style="margin-top:1rem">Uploads are sent to the app server for inference. Temporary processing files are removed after analysis. Avoid uploading private footage.</div>', unsafe_allow_html=True)

    if uploaded_file is not None and st.session_state.get("analysis_error"):
        st.error(f"Analysis failed: {st.session_state.analysis_error}")

    result = st.session_state.get("analysis_result")
    if result and uploaded_file is not None and st.session_state.get("analysis_filename") == uploaded_file.name:
        st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
        st.subheader("Analysis result")
        result_col, scores_col = st.columns([.9, 1.1])
        with result_col:
            if result["label"] == "Fake":
                st.error("Model prediction: Fake")
            else:
                st.success("Model prediction: Real")
            st.metric("Processing time", f"{result['elapsed']:.1f} sec")
            st.caption("This is a model output, not a guarantee about the video's authenticity.")
        with scores_col:
            st.markdown("**Class scores**")
            st.markdown(f"Real score: **{result['real_score']:.1%}**")
            st.progress(result["real_score"])
            st.markdown(f"Fake score: **{result['fake_score']:.1%}**")
            st.progress(result["fake_score"])
            st.caption("Softmax scores are not calibrated probabilities.")

        st.markdown("**Frames shown from the analyzed sequence**")
        frame_columns = st.columns(4)
        frame_numbers = np.linspace(1, FRAME_COUNT, 4, dtype=int)
        for col, frame, frame_number in zip(frame_columns, result["sample_frames"], frame_numbers):
            with col:
                st.image(frame, caption=f"Frame {frame_number}", use_container_width=True)

st.markdown('<div class="footer">Deepfake video detection research prototype. ResNeXt-50 + LSTM. <a href="https://doi.org/10.1109/ICSIT65336.2025.11294356">Read the published paper</a></div>', unsafe_allow_html=True)
