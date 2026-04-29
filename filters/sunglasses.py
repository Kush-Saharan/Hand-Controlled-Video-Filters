import cv2
import numpy as np
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

sunglasses = cv2.imread("assets/sunglasses.png", cv2.IMREAD_UNCHANGED)

def overlay_image(bg, overlay, x, y):
    oh, ow = overlay.shape[:2]
    fh, fw = bg.shape[:2]

    # clip to visible region instead of early return
    x1_bg = max(x, 0);       y1_bg = max(y, 0)
    x2_bg = min(x + ow, fw); y2_bg = min(y + oh, fh)
    if x2_bg <= x1_bg or y2_bg <= y1_bg:
        return bg

    x1_ov = x1_bg - x; y1_ov = y1_bg - y
    x2_ov = x1_ov + (x2_bg - x1_bg)
    y2_ov = y1_ov + (y2_bg - y1_bg)

    roi    = overlay[y1_ov:y2_ov, x1_ov:x2_ov]
    alpha  = roi[:, :, 3:4] / 255.0
    bg_roi = bg[y1_bg:y2_bg, x1_bg:x2_bg].astype(np.float32)
    ov_rgb = roi[:, :, :3].astype(np.float32)

    blended = alpha * ov_rgb + (1 - alpha) * bg_roi
    bg[y1_bg:y2_bg, x1_bg:x2_bg] = blended.astype(np.uint8)
    return bg

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # expand canvas so rotated corners don't get clipped
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(
        image, matrix, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )
    return rotated

def apply(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            left_eye  = face.landmark[33]
            right_eye = face.landmark[263]

            x1, y1 = int(left_eye.x  * w), int(left_eye.y  * h)
            x2, y2 = int(right_eye.x * w), int(right_eye.y * h)

            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

            cx, cy   = (x1 + x2) // 2, (y1 + y2) // 2
            eye_dist = int(np.hypot(x2 - x1, y2 - y1))

            glass_width  = int(eye_dist * 2.2)
            glass_height = int(glass_width * 0.5)

            # resize first
            resized = cv2.resize(sunglasses, (glass_width, glass_height))

            # rotate based on face tilt
            rotated = rotate_image(resized, angle)

            # re-centre using rotated canvas size (larger after rotation)
            rh, rw = rotated.shape[:2]
            x = cx - rw // 2
            y = cy - rh // 2

            frame = overlay_image(frame, rotated, x, y)

    return frame