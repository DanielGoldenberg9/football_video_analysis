import sys

sys.path.insert(0, "../")
from utils import get_center_bbox, mesure_distance


class PlayerBallAssigner:
    def __init__(self) -> None:
        self.max_player_ball_distance = 80

    def assign_ball_to_player(self, players, ball_bbox):
        ball_pos = get_center_bbox(ball_bbox)
        min_distance = float("inf")
        assign_player = -1
        for player_id, player in players.items():
            player_bbox = player["bbox"]
            distance_left = mesure_distance((player_bbox[0], player_bbox[-1]), ball_pos)
            distance_right = mesure_distance(
                (player_bbox[2], player_bbox[-1]), ball_pos
            )
            distance = min(distance_left, distance_right)
            if distance < self.max_player_ball_distance and distance < min_distance:
                min_distance = distance
                assign_player = player_id
        return assign_player
