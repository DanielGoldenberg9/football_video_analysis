from typing import NamedTuple, Optional, Union
import numpy as np


def get_center_bbox(bbox):
    """
    Returns the center of the bounding box
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return int(center_x), int(center_y)


def bbox_to_int(bbox):
    """
    Returns the bounding box in integer format
    """
    x1, y1, x2, y2 = bbox
    return int(x1), int(y1), int(x2), int(y2)


def get_bbox_width(bbox) -> int:
    return bbox[2] - bbox[0]


def mesure_distance(point1, point2) -> float:
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def check_if_obj_has_bbox(obj: Union[object, None]) -> Union[NamedTuple, None]:
    return None if obj is None else obj.bbox
