from dataclasses import dataclass, field
import numpy as np
import cv2


@dataclass
class BallControlBoard:
    control: dict[int:float] = field(default_factory=dict)

    def draw_annotation(self, frame):

        data = frame.data.copy()
        surface = data.copy()
        shape_frame = data.shape[:2]
        positions = [
            (int(shape_frame[1] * 0.75), int(shape_frame[0] * 0.85)),
            shape_frame[::-1],
        ]
        alpha = 0.5
        cv2.rectangle(surface, positions[0], positions[1], (255, 255, 255), -1)
        cv2.addWeighted(surface, alpha, data, 1 - alpha, 0, data)
        delta_y = int(positions[1][1] - positions[0][1]) // 3
        padd = 10
        cv2.putText(
            data,
            "team ball control",
            (positions[0][0] + padd, positions[0][1] + delta_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            data,
            f"team 1 = {self.control[0][frame.frame_num]:.3f}%",
            (positions[0][0] + padd, positions[0][1] + (2 * delta_y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            data,
            f"team 2 = {self.control[1][frame.frame_num]:.3f}%",
            (positions[0][0] + padd, positions[1][1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )
        return data
