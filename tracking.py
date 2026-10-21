"""YuNet face boxes with an offline Haar fallback."""
from pathlib import Path
import urllib.request

import cv2
import numpy as np


def make_detector():
    path = Path(__file__).with_name("face_yunet.onnx")
    if not path.exists():
        temporary = path.with_suffix(".download")
        try:
            print("Baixando modelo YuNet…")
            with urllib.request.urlopen(
                "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
                "face_detection_yunet_2023mar.onnx", timeout=30
            ) as response:
                temporary.write_bytes(response.read())
            temporary.replace(path)
        except OSError:
            temporary.unlink(missing_ok=True)
            print("Sem download disponível; usando Haar.")
    if path.exists():
        try:
            return cv2.FaceDetectorYN_create(str(path), "", (200, 150)), None
        except cv2.error:
            print("Modelo YuNet inválido; usando Haar.")
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise RuntimeError("Detector indisponível. Reinstale requirements.txt.")
    return None, cascade


def detect(frame, yunet, cascade):
    """Uniform floating-point boxes for both detectors, in frame coordinates."""
    if yunet is not None:
        yunet.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = yunet.detect(frame)
        return np.empty((0, 4)) if faces is None else np.asarray(faces[:, :4], dtype=float)
    gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    faces = cascade.detectMultiScale(gray, 1.15, 5, minSize=(28, 28))
    return np.asarray(faces, dtype=float).reshape(-1, 4)
