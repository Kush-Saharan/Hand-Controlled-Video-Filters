import cv2
import numpy as np

def apply(frame, val=50, thresh=50):
    """
    TV static / noise effect
    val: intensity of noise (0–255)
    thresh: probability of noise (0–100)
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    noisy = gray.copy()

    height, width = gray.shape

    # vectorized random masks (FASTER than loops)
    noise_mask = np.random.randint(0, 100, (height, width)) < thresh
    add_mask = np.random.randint(0, 2, (height, width)) == 0

    random_vals = np.random.randint(0, val + 1, (height, width))

    # apply noise
    noisy = noisy.astype(np.int16)

    noisy[noise_mask & add_mask] += random_vals[noise_mask & add_mask]
    noisy[noise_mask & ~add_mask] -= random_vals[noise_mask & ~add_mask]

    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return noisy