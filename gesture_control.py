import cv2

from hand_tracking import HandTracker
from utils import draw_boxes, get_selection

from filters import brightness, sepia, emboss, duo_tone, tv_60


# ---------------- FILTER REGISTRY ----------------
FILTERS = [
    ("None", lambda f: f),
    ("Sepia", sepia.apply),
    ("Emboss", lambda f: emboss.apply(f, size=5, direction=1)),
    ("Bright", lambda f: brightness.apply(f, 1.3)),
    ("DuoTone", lambda f: duo_tone.apply(f, 1.3, 2, 1, "dark")),
    ("TV", lambda f: tv_60.apply(f, 80, 40)),
]


# ---------------- INIT ----------------
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

tracker = HandTracker()

current_filter = 0
hover_index = None
hover_frames = 0
HOVER_THRESHOLD = 12  # frames (~0.4 sec)


# ---------------- LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Draw UI
    names = [f[0] for f in FILTERS]
    boxes = draw_boxes(frame, names)

    # Hand tracking
    finger = tracker.get_finger(frame)

    if finger:
        cv2.circle(frame, finger, 10, (0, 255, 0), -1)

    selected = get_selection(finger, boxes)

    # -------- Hover selection (debounce) --------
    if selected is not None:
        if hover_index == selected:
            hover_frames += 1
        else:
            hover_index = selected
            hover_frames = 0

        if hover_frames > HOVER_THRESHOLD:
            current_filter = selected
    else:
        hover_index = None
        hover_frames = 0

    # -------- Apply filter --------
    frame = FILTERS[current_filter][1](frame)

    # -------- Highlight selected --------
    if hover_index is not None:
        x1, y1, x2, y2 = boxes[hover_index]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.imshow("Gesture Control Filters", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()