from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import namedtuple
import numpy as np
import sys
import cv2
from sklearn.cluster import KMeans

sys.path.insert(0, "../")
from utils.bbox_utils import get_center_bbox, get_bbox_width, bbox_to_int
from utils.shapes_utils import draw_ellipse, draw_triangle, draw_rectangle

bbox_class = namedtuple("bbox", ["x1", "y1", "x2", "y2"])


class TrackingObject(ABC):

    @abstractmethod
    def draw_annotation(self, frame: np.ndarray) -> np.ndarray:
        """draw annotation on the frame"""


class Player(TrackingObject):
    def __init__(self, bbox, _id) -> None:
        self.bbox = bbox_class(*bbox)
        self._id = _id
        self.team_color = None
        self.team = None
        self.has_ball = False

    def draw_annotation(self, frame: np.ndarray) -> np.ndarray:

        frame = draw_ellipse(frame, self.bbox, self.team_color)
        frame = draw_rectangle(frame, self.bbox, self.team_color, self._id)
        if self.has_ball:
            frame = draw_triangle(frame, self.bbox, (0, 255, 0))
        return frame


class Referee(TrackingObject):
    """
    This class represents a Referee in the game.
    """

    def __init__(self, bbox, _id) -> None:
        self._id = _id
        self.bbox = bbox_class(*bbox)
        self.color = (0, 255, 255)

    def draw_annotation(self, frame: np.ndarray) -> np.ndarray:
        frame = draw_ellipse(frame, self.bbox, self.color)
        return frame


class Ball(TrackingObject):
    """
    This class represents a Ball in the game.
    """

    def __init__(self, bbox) -> None:
        self.bbox = bbox_class(*bbox)
        self.color: tuple[int, int, int] = (0, 0, 255)

    def draw_annotation(self, frame):
        frame = draw_triangle(frame, self.bbox, self.color)
        return frame

    def update_bbox(self, bbox: list[float]):
        self.bbox = bbox_class(*bbox_to_int(bbox))
