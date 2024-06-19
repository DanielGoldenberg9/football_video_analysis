import numpy as np
import pandas as pd


def detect_player_label(labels: np.ndarray) -> int:
    corners = [
        labels[0, 0],
        labels[0, -1],
        labels[-1, 0],
        labels[-1, -1],
    ]
    corner_label = max(set(corners), key=corners.count)
    return 1 - corner_label


def insert_values_by_indices_to_list(
    l: np.ndarray,
) -> list:
    df = pd.DataFrame(l)
    df.ffill(inplace=True)
    df[df.isna()] = 0
    return df.to_numpy().squeeze().tolist()
