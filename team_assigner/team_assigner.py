from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team = {}

    def get_player_color(self, frame, bbox):
        # cropping
        image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
        top_half_img = image[: image.shape[0] // 2, :]

        croped_2D = top_half_img.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=5).fit(croped_2D)
        labels = kmeans.labels_.reshape(top_half_img.shape[:2])
        corners_cluster = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
        backround_label_color = max(set(corners_cluster), key=corners_cluster.count)
        player_label = 1 - backround_label_color
        return kmeans.cluster_centers_[player_label]

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player in player_detections.items():
            bbox = player["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
        self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=5).fit(
            player_colors
        )
        self.team_colors[0] = self.kmeans.cluster_centers_[0]
        self.team_colors[1] = self.kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team.keys():
            return self.player_team[player_id]
        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        self.player_team[player_id] = team_id
        return team_id
