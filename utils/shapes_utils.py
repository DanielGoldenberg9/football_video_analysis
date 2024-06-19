import cv2
import numpy as np
from utils import get_bbox_width, get_center_bbox


def draw_ellipse(frame, bbox, color):
    x_center, _ = get_center_bbox(bbox)
    y2 = int(bbox[3])
    width = get_bbox_width(bbox)
    cv2.ellipse(
        frame,
        center=(x_center, y2),
        axes=(int(width), int(0.4 * width)),
        angle=0.0,
        startAngle=-40,
        endAngle=240,
        color=color,
        thickness=2,
    )
    return frame


def draw_rectangle(frame, bbox, color, track_id=None):
    rect_width, rect_height = 40, 27
    x_center, _ = get_center_bbox(bbox)
    y2 = int(bbox[3])
    cv2.rectangle(
        frame,
        (x_center - rect_width // 2, y2),
        (x_center + rect_width // 2, y2 + (rect_height // 2) + 6),
        color=color,
        thickness=-1,
    )
    cv2.putText(
        frame,
        str(track_id),
        (x_center, y2 + 25),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(0, 0, 0),
        thickness=2,
    )

    return frame


def draw_triangle(frame, bbox, color):
    paddingx = 10
    paddingy = 15
    y = int(bbox[1])
    x_center, _ = get_center_bbox(bbox)
    triangle_points = np.array(
        [
            [x_center, y],
            [x_center + paddingx, y - paddingy],
            [x_center - paddingx, y - paddingy],
        ]
    )
    cv2.drawContours(
        frame,
        [triangle_points],
        0,
        color=color,
        thickness=cv2.FILLED,
    )
    return frame
