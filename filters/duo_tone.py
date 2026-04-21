import cv2
import numpy as np

def _exp_lut(exp):
    """Precompute LUT for exponential mapping"""
    table = np.array([min((i ** exp), 255) for i in range(256)], dtype=np.uint8)
    return table


def apply(frame, exp=1.2, ch1=2, ch2=1, mode="dark"):
    """
    Duo tone filter (optimized + reusable)

    exp: intensity (1.0–2.0)
    ch1, ch2: channels to enhance (0=B, 1=G, 2=R)
    mode: "dark" or "light"
    """

    res = frame.copy()

    lut_main = _exp_lut(exp)
    lut_alt = _exp_lut(2 - exp)

    for i in range(3):
        if i == ch1 or i == ch2:
            res[:, :, i] = cv2.LUT(res[:, :, i], lut_main)
        else:
            if mode == "light":
                res[:, :, i] = cv2.LUT(res[:, :, i], lut_alt)
            else:
                res[:, :, i] = 0

    return res