import cv2
import numpy as np

def apply(frame, val=1.0):
    """
    Brightness + saturation control
    val: 0.0 → dark, 1.0 → normal, >1.0 → brighter (recommended range: 0.5–1.5)
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

    # scale saturation and brightness
    hsv[:, :, 1] *= val   # Saturation
    hsv[:, :, 2] *= val   # Value (brightness)

    # clip values
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

    hsv = hsv.astype(np.uint8)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return result