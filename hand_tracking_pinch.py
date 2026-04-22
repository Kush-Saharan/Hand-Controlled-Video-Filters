import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands


class HandTracker:
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def get_hand_data(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]

            # index finger tip (8)
            ix = int(hand.landmark[8].x * w)
            iy = int(hand.landmark[8].y * h)

            # thumb tip (4)
            tx = int(hand.landmark[4].x * w)
            ty = int(hand.landmark[4].y * h)

            # distance
            dist = math.hypot(ix - tx, iy - ty)

            return {
                "index": (ix, iy),
                "thumb": (tx, ty),
                "distance": dist
            }

        return None