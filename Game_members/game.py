from dataclasses import dataclass, field
from .members import Player, Referee, Ball
from .team import TeamAssigner
from .ball_control_board import BallControlBoard


import numpy as np
import pandas as pd
from itertools import chain
from functools import lru_cache
import sys

sys.path.insert(0, "../")
from utils import (
    check_if_obj_has_bbox,
    bbox_to_int,
    get_center_bbox,
    mesure_distance,
    insert_values_by_indices_to_list,
)


@dataclass
class Frame:
    frame_num: int = 0
    data: np.ndarray = field(default=None, init=True)
    players: dict[Player] = field(default_factory=dict, init=False)
    referees: dict[Referee] = field(default_factory=dict, init=False)
    ball: Ball = field(default=None, init=False)
    player_with_ball: Player = field(default=None, init=False)

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        if self.ball is not None:
            ball = [self.ball]
        else:
            ball = []
        return iter(
            list(chain(self.players.values()))
            + list(chain(self.referees.values()))
            + ball
        )

    def assign_ball_to_player(self):
        max_distance = 60
        closest_player = None
        closest_distance = float("inf")
        ball_pos = get_center_bbox(self.ball.bbox)
        for player in self.players.values():
            dis_left_foot_ball = mesure_distance(
                (player.bbox.x1, player.bbox.y2), ball_pos
            )
            dis_right_foot_ball = mesure_distance(
                (player.bbox.x2, player.bbox.y2), ball_pos
            )
            distance = min(dis_right_foot_ball, dis_left_foot_ball)
            if distance < max_distance and distance < closest_distance:
                closest_player = player
                closest_distance = distance

        if closest_player is not None:
            closest_player.has_ball = True
            self.player_with_ball = closest_player
            return closest_player.team


class Game:
    def __init__(self):
        self.frames: list[Frame] = []
        self.team_assigner = TeamAssigner()
        self.ball_control_board = BallControlBoard()

    def add_frame(self, frame):
        self.frames.append(Frame(frame_num=len(self.frames), data=frame))

    @lru_cache(maxsize=255)
    def get_all_members(self, members: str):
        # list comprehension is faster and more redable than for loop.
        if members == "ball":
            return [frame[members] for frame in self.frames]
        return [member for frame in self.frames for member in frame[members]]

    def __getitem__(self, key):
        return self.frames[key]

    def __iter__(self):
        return zip(range(len(self.frames)), self.frames)

    def interpolate_ball_positions(self):
        all_balls = self.get_all_members("ball")
        all_balls_bboxs = [[None] * 4 if b is None else b.bbox for b in all_balls]
        df_balls = pd.DataFrame(all_balls_bboxs, columns=["x1", "y1", "x2", "y2"])
        df_balls.interpolate(method="linear", inplace=True)
        df_balls.bfill(
            inplace=True
        )  # if first frame ball is not detected,fill with closest value
        list_of_all_balls = (
            df_balls.to_numpy().tolist()
        )  # restore to regular format shape(frame_num,4)

        for frame_num, frame in self:
            if isinstance(frame.ball, Ball):
                frame.ball.update_bbox(list_of_all_balls[frame_num])
            else:
                frame.ball = Ball(bbox=bbox_to_int(list_of_all_balls[frame_num]))

    def assign_teams(self):
        self.team_assigner.assign_team(self.frames[0])
        for frame in self.frames:
            for player in frame.players.values():
                self.team_assigner.set_player_team(frame, player)

    def ball_controler(self):
        team_ball_contrall = {0: [], 1: []}
        bad_indices = []
        good_indices = []
        for frame in self.frames:

            team = (
                frame.assign_ball_to_player()
            )  # assign to player a ball control property
            if team is not None:
                team_ball_contrall[team].append(1)
                team_ball_contrall[1 - team].append(0)
                good_indices.append(frame.frame_num)
            else:
                bad_indices.append(frame.frame_num)
        all_indices = good_indices + bad_indices
        sort_indices_byframes = np.argsort(all_indices)

        for t in team_ball_contrall:
            team_ball_contrall[t] = (
                np.cumsum(team_ball_contrall[t])
                / range(1, len(team_ball_contrall[t]) + 1)
                * 100
            )  # percentage of ball control by a team in each frame
            team_ball_contrall[t] = np.append(
                team_ball_contrall[t], [None] * len(bad_indices)
            )  # add bad frames indices
            team_ball_contrall[t] = insert_values_by_indices_to_list(
                team_ball_contrall[t][sort_indices_byframes]
            )  # initerpolate bad frames
        self.ball_control_board.control = team_ball_contrall
