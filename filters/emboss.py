import cv2
import numpy as np


def _kernel_generator(size):
    kernel = np.zeros((size, size), dtype=np.int8)

    for i in range(size):
        for j in range(size):
            if i < j:
                kernel[i, j] = -1
            elif i > j:
                kernel[i, j] = 1

    return kernel


def apply(frame, size=3, direction=0):
    """
    Emboss filter

    size: kernel size (3–9 recommended)
    direction: 0=BL, 1=BR, 2=TR, 3=TL
    """

    # ensure valid size
    size = max(3, size)
    if size % 2 == 0:
        size += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kernel = _kernel_generator(size)
    kernel = np.rot90(kernel, direction)

    embossed = cv2.filter2D(gray, -1, kernel)

    # add offset for visibility
    embossed = cv2.add(embossed, np.full_like(embossed, 128))

    # convert back to 3-channel (important for pipeline consistency)
    embossed = cv2.cvtColor(embossed, cv2.COLOR_GRAY2BGR)

    return embossed