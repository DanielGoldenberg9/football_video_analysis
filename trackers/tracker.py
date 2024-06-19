from ultralytics import YOLO
import cv2
import pandas as pd
import supervision as sv
import numpy as np
import pickle
import os, sys

sys.path.insert(0, "../")
from utils.bbox_utils import get_center_bbox, get_bbox_width, bbox_to_int
from Game_members import Game, Frame, Player, Referee, Ball


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        # for memory issues set batch size
        # set conf 0.1, it works generally well -> hyperparameter
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_detection = self.model.predict(frames[i : i + batch_size], conf=0.1)
            detections += batch_detection
            if i > 81:
                break
        return detections

    def _convert_goalkeeper_to_player(
        self,
        detections: sv.Detections,
        cls_names: dict[str, int],
        cls_names_inv: dict[int, str],
    ):
        for object_id, class_indx in enumerate(detections.class_id):
            if cls_names[class_indx] == "Goalkeeper":
                detections.class_id[object_id] = cls_names_inv["Player"]

    def get_object_tracks(
        self,
        frames: list[np.ndarray],
        read_from_stub: bool = False,
        stub_path: str = None,
    ) -> dict[str, list[dict[int, dict[str, list]]]]:

        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks

        detections: list["ultralytics.boxes"] = self.detect_frames(frames)

        tracks = Game()
        # axample tracks ->{players:[{0:{"bbox":[x1,y1,x2,y2]},{1:{"bbox":[x1,y1,x2,y2]}...{11:{"bbox":[x1,y1,x2,y2]}...},
        #                            {0:{"bbox":[x1,y1,x2,y2]},{1:{"bbox":[x1,y1,x2,y2]}...{11:{"bbox":[x1,y1,x2,y2]}...}]}

        for frame_indx, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {value: key for key, value in cls_names.items()}
            # convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # because of small dataset, convert goalkeeper to player object
            self._convert_goalkeeper_to_player(
                detection_supervision, cls_names, cls_names_inv
            )

            # track objects
            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision
            )

            tracks.add_frame(frames[frame_indx])

            # detection_with_tracks of form [cords,mask,confidence,class_id,tracker_id]

            for frame_detection in detection_with_tracks:

                bbox = bbox_to_int(frame_detection[0].tolist())
                cls_id = frame_detection[3]
                tracker_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks[frame_indx]["players"][tracker_id] = Player(
                        bbox=bbox, _id=tracker_id
                    )

                if cls_id == cls_names_inv["referee"]:
                    tracks[frame_indx]["referees"][tracker_id] = Referee(
                        bbox=bbox, _id=tracker_id
                    )
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == cls_names_inv["ball"]:
                    tracks[frame_indx].ball = Ball(bbox=bbox)

        if stub_path:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)
        self.tracks = tracks
        return tracks

    def adding_game_properties(self):
        self.tracks.interpolate_ball_positions()
        self.tracks.assign_teams()
        self.tracks.ball_controler()

    def draw_annotations(self) -> np.ndarray:
        output_frames = []
        for _, frame in self.tracks:
            for obj in frame:
                frame.data = obj.draw_annotation(frame.data)
            frame.data = self.tracks.ball_control_board.draw_annotation(frame)
            output_frames.append(frame.data)
        return output_frames
