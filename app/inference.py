import os
import torch
import cv2
import dlib
import gdown
import numpy as np
import logging
from model_def import CrossModalFakeVideoDetector

logger = logging.getLogger(__name__)

# === Step 1: Auto-download model & .dat file if not exists ===
def download_weights():
    model_path = os.path.join(os.path.dirname(__file__), "model.pth")
    dat_path = os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat")

    if not os.path.exists(model_path):
        logger.info("Downloading model.pth from Google Drive...")
        gdown.download(id="1MTZRaA508cf6Zl501Qa1qtnOizxfI7R_", output=model_path, quiet=False)

    if not os.path.exists(dat_path):
        logger.info("Downloading shape_predictor_68_face_landmarks.dat from Google Drive...")
        gdown.download(id="1flCwl_98oPXLNOk1cE_hK9Ui6_XWZvyF", output=dat_path, quiet=False)

download_weights()

# === Step 2: Load predictor ===
predictor_path = os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat")
try:
    predictor = dlib.shape_predictor(predictor_path)
    logger.info("✓ Face landmark predictor loaded")
except Exception as e:
    logger.warning(f"⚠ Face landmark predictor not available: {e}")
    predictor = None

# === Step 3: Load the deepfake detection model ===
def load_model():
    model = CrossModalFakeVideoDetector()
    model_path = os.path.join(os.path.dirname(__file__), "model.pth")

    try:
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None

        state_dict = torch.load(model_path, map_location="cpu")
        
        # Adjust if loaded state_dict has unexpected structure
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        model.load_state_dict(state_dict, strict=False)
        model.eval()
        logger.info("✓ Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"⚠ Model loading issue: {e}")
        return None

# === Step 4: Predict function ===
def predict_video(video_file_path, model):
    try:
        if not os.path.exists(video_file_path):
            return {"error": "Video file not found", "prediction": None}

        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            return {"error": "Invalid video file", "prediction": None}

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if frame_count < 10:
            return {"error": "Video too short", "prediction": None}

        logger.info(f"Processed video: {frame_count} frames, {fps} FPS")

        # === Mock prediction - replace with actual inference ===
        return {
            "prediction": "real",  # Replace with `model.predict(...)` logic
            "confidence": 0.95,
            "frames_analyzed": frame_count,
            "fps": fps
        }

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return {"error": str(e), "prediction": None}
