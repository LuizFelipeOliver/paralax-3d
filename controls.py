"""Face position to camera orbit, with the original lightweight smoothing."""
import numpy as np

YAW_GAIN = 1.57
GAIN_Y = 7.0             # Larger vertical orbit exposes the top and bottom faces.
CAM_Z0 = 7.0
REF_FACE = 0.24
KZ = 14.0
Z_CLAMP = (4.0, 11.0)
EMA = 0.35


def face_position(box, width, height):
    x, y, w, h = np.asarray(box, dtype=float)[:4]
    return np.array([(x + w / 2 - width / 2) / (width / 2),
                     (height / 2 - y - h / 2) / (height / 2), w / width])


def smooth_position(previous, measured):
    return previous + EMA * (measured - previous)


def head_to_cam(dx, dy, wfrac):
    return (float(np.clip(dx * YAW_GAIN, -YAW_GAIN, YAW_GAIN)),
            float(np.clip(dy * GAIN_Y, -GAIN_Y, GAIN_Y)),
            float(np.clip(CAM_Z0 + (REF_FACE - wfrac) * KZ, *Z_CLAMP)))
