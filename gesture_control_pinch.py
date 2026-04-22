import cv2

from hand_tracking_pinch import HandTracker
# from utils import draw_boxes, get_selection
from filters import brightness, sepia, emboss, duo_tone, tv_60


FILTERS = [
    ("None", lambda f: f),
    ("Sepia", sepia.apply),
    ("Emboss", lambda f: emboss.apply(f, size=5, direction=1)),
    ("Bright", lambda f: brightness.apply(f, 1.3)),
    ("DuoTone", lambda f: duo_tone.apply(f, 1.3, 2, 1, "dark")),
    ("TV", lambda f: tv_60.apply(f, 80, 40)),
]


cap = cv2.VideoCapture(0)
tracker = HandTracker()

current_filter = 0
PINCH_THRESHOLD = 40   # distance (tune this!)
pinch_active = False


while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    

    # names = [f[0] for f in FILTERS]
    # boxes = draw_boxes(frame, names)

    data = tracker.get_hand_data(frame)

    if data:
        ix, iy = data["index"]
        tx, ty = data["thumb"]
        dist = data["distance"]

        # draw points
        cv2.circle(frame, (ix, iy), 8, (0,255,0), -1)
        cv2.circle(frame, (tx, ty), 8, (255,0,0), -1)
        cv2.line(frame, (ix, iy), (tx, ty), (255,255,255), 2)

        # -------- PINCH DETECTION --------
        if dist < PINCH_THRESHOLD:
            if not pinch_active:
                pinch_active = True

                current_filter = (current_filter + 1) % len(FILTERS)
        else:
            pinch_active = False

    # apply filter
    frame = FILTERS[current_filter][1](frame)

    # show current filter name
    filter_name = FILTERS[current_filter][0]
    cv2.putText(frame, f"Filter: {filter_name}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2)

    cv2.imshow("Pinch Gesture Filters", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()