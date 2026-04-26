import cv2
import numpy as np

def _gamma_lut(gamma):
    """Safer LUT using gamma correction"""
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
    ]).astype("uint8")
    return table


def apply(frame, exp=1.2, ch1=2, ch2=1, mode="light", blend=0.6):
    """
    Improved Duo tone filter

    exp: intensity (1.0–2.0)
    ch1, ch2: channels to enhance (0=B,1=G,2=R)
    mode: "light" or "dark"
    blend: mix factor (0–1)
    """

    res = frame.copy()

    # safer LUTs
    lut_main = _gamma_lut(exp)
    lut_alt = _gamma_lut(max(0.5, 2 - exp))

    for i in range(3):
        channel = res[:, :, i]

        if i == ch1 or i == ch2:
            enhanced = cv2.LUT(channel, lut_main)
        else:
            if mode == "light":
                enhanced = cv2.LUT(channel, lut_alt)
            else:
                enhanced = np.zeros_like(channel)

        res[:, :, i] = cv2.addWeighted(channel, 1 - blend, enhanced, blend, 0)

    return res