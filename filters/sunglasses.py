import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# load sunglasses PNG (with alpha channel)
sunglasses = cv2.imread("assets/sunglasses.png", cv2.IMREAD_UNCHANGED)


def overlay_image(bg, overlay, x, y, w, h):
    overlay = cv2.resize(overlay, (w, h))

    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
            bg[y:y+h, x:x+w, c] = (
                alpha * overlay[:, :, c] +
                (1 - alpha) * bg[y:y+h, x:x+w, c]
            )
    return bg


def apply(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:

            left_eye = face.landmark[33]
            right_eye = face.landmark[263]

            x1, y1 = int(left_eye.x * w), int(left_eye.y * h)
            x2, y2 = int(right_eye.x * w), int(right_eye.y * h)

            # center
            cx, cy = (x1 + x2)//2, (y1 + y2)//2

            # width based on eye distance
            eye_dist = int(np.hypot(x2 - x1, y2 - y1))
            glass_width = int(eye_dist * 2.2)
            glass_height = int(glass_width * 0.5)

            # top-left position
            x = int(cx - glass_width / 2)
            y = int(cy - glass_height / 2)

            # boundary check
            if x < 0 or y < 0 or x + glass_width > w or y + glass_height > h:
                return frame

            frame = overlay_image(frame, sunglasses, x, y, glass_width, glass_height)

    return frame