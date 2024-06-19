from sklearn.cluster import KMeans
import numpy as np
import sys

sys.path.insert(0, "../")
from utils.team_utils import detect_player_label


class TeamAssigner:
    def __init__(self):
        self.players_team: dict = {}
        self.color_teams: dict = {}

    def get_shirt_color(self, frame: np.ndarray, player):
        image = frame[player.bbox.y1 : player.bbox.y2, player.bbox.x1 : player.bbox.x2]
        croped = image[: 2 * image.shape[0] // 3, :]
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(
            croped.reshape(-1, 3)  # 3 color channels features
        )
        labels = kmeans.labels_
        labels = labels.reshape(croped.shape[:2])
        player_label = detect_player_label(labels)
        return kmeans.cluster_centers_[player_label]

    def get_all_players_color(self, frame):
        shirt_colors = []
        for player in frame.players.values():
            player_shirt_color = self.get_shirt_color(frame.data, player)
            shirt_colors.append(player_shirt_color)
        return shirt_colors

    def assign_team(self, frame):
        shirt_colors = self.get_all_players_color(frame)
        self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(shirt_colors)
        self.color_teams[0] = self.kmeans.cluster_centers_[0]
        self.color_teams[1] = self.kmeans.cluster_centers_[1]

    def set_player_team(self, frame, player):
        if player._id not in self.players_team:
            shirt_color = self.get_shirt_color(frame.data, player)
            pred = self.kmeans.predict(shirt_color.reshape(1, -1))[0]
            self.players_team[player._id] = pred

        player.team = self.players_team[player._id]
        player.team_color = self.color_teams[player.team]
