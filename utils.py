import cv2

def draw_boxes(frame, filter_names):
    boxes = []
    h, w, _ = frame.shape

    box_w = w // len(filter_names)
    box_h = 80

    for i, name in enumerate(filter_names):
        x1 = i * box_w
        y1 = 0
        x2 = x1 + box_w
        y2 = box_h

        boxes.append((x1, y1, x2, y2))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, name, (x1 + 10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    return boxes


def get_selection(finger, boxes):
    if finger is None:
        return None

    x, y = finger

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        if x1 < x < x2 and y1 < y < y2:
            return i

    return None