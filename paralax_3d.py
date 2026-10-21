#!/usr/bin/env python3
"""Webcam-driven cube perspective. Run ./run.sh; F fullscreen, Q/Esc exit."""
from pathlib import Path
import argparse
import sys
import unittest

import cv2
import numpy as np

from controls import REF_FACE, face_position, head_to_cam, smooth_position
from scene import W, H, make_bg, render_scene
from tracking import detect, make_detector
from wall_scene import render_wall

WINDOW = "paralax-3d"


def set_fullscreen(enabled):
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL)


def selftest():
    suite = unittest.defaultTestLoader.discover(
        str(Path(__file__).parent / "tests"), pattern="test_python.py")
    if not unittest.TextTestRunner().run(suite).wasSuccessful():
        raise SystemExit(1)


def main(argv):
    parser = argparse.ArgumentParser(description="Perspectiva 3D controlada pela webcam")
    parser.add_argument("--scene", choices=("cubo", "parede"), default="cubo")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args(argv[1:])
    background = make_bg()
    if args.test:
        selftest()
        output = Path("/tmp/opencode")
        output.mkdir(parents=True, exist_ok=True)
        for name, theta, height, distance in [
            ("center", 0, 0, 7), ("left", 1.25, 0.2, 7),
            ("right", -1.25, 0.2, 7), ("left90", 1.57, 0.2, 7),
            ("up", 0.1, 5, 7), ("down", 0.1, -5, 7),
        ]:
            path = output / f"paralax_{name}.png"
            if not cv2.imwrite(str(path), render_scene(theta, height, distance, background)):
                raise OSError(f"Não foi possível salvar {path}")
            print("ok:", path)
        for name, dx in (("center", 0), ("left", -0.65), ("right", 0.65)):
            path = output / f"paralax_wall_{name}.png"
            if not cv2.imwrite(str(path), render_wall(dx, 0, REF_FACE, background)):
                raise OSError(f"Não foi possível salvar {path}")
            print("ok:", path)
        return

    yunet, cascade = make_detector()
    cap = cv2.VideoCapture(0)
    try:
        if not cap.isOpened():
            raise SystemExit("Não foi possível abrir a webcam. Feche abas e apps que usam a câmera.")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, W, H)
        fullscreen = args.fullscreen
        set_fullscreen(fullscreen)
        position = np.array([0.0, 0.0, REF_FACE])
        misses = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                misses += 1
                if misses > 30:
                    raise SystemExit("A webcam parou de enviar imagens.")
                if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
                    break
                continue
            misses = 0
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            small_width = 200
            small_height = max(1, round(small_width * height / width))
            small = cv2.resize(frame, (small_width, small_height))
            faces = detect(small, yunet, cascade)
            if len(faces):
                box = max(faces, key=lambda f: f[2] * f[3])
                position = smooth_position(position, face_position(box, small_width, small_height))
            if args.scene == "parede":
                image = render_wall(*position, background)
            else:
                image = render_scene(*head_to_cam(*position), background)
            image[10:190, W - 250:W - 10] = cv2.resize(frame, (240, 180))
            cv2.imshow(WINDOW, image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("f"):
                fullscreen = not fullscreen
                set_fullscreen(fullscreen)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv)
