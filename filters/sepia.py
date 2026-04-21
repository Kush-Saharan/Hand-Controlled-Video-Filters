import cv2
import numpy as np


def apply(frame):
    """
    Sepia filter
    """

    # convert to float for precision
    img = frame.astype(np.float32)

    # sepia transformation matrix (BGR directly)
    kernel = np.array([
        [0.131, 0.534, 0.272],
        [0.168, 0.686, 0.349],
        [0.189, 0.769, 0.393]
    ])

    sepia = cv2.transform(img, kernel)

    # clip values and convert back
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)

    return sepia