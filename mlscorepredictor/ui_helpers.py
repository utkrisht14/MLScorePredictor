from __future__ import annotations

import pandas as pd


def round_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rounded = frame.copy()
    for column in rounded.select_dtypes(include=["float"]).columns:
        rounded[column] = rounded[column].round(3)
    return rounded


def player_impact_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in frame.to_dict("records"):
        if row["position"] == "GK":
            impact = max(row["gk"] - 70, 0) * 0.02
        elif row["position"] == "DF":
            impact = (row["defending"] * 0.45 + row["physical"] * 0.25 + row["overall"] * 0.30) / 100
        elif row["position"] == "MF":
            impact = (row["passing"] * 0.40 + row["dribbling"] * 0.25 + row["overall"] * 0.35) / 100
        else:
            impact = (row["shooting"] * 0.45 + row["pace"] * 0.25 + row["overall"] * 0.30) / 100
        rows.append(
            {
                "Player": row["player_name"],
                "Team": row["team"],
                "Position": row["position"],
                "Overall": row["overall"],
                "Impact index": round(impact, 3),
            }
        )
    return pd.DataFrame(rows).sort_values("Impact index", ascending=False)
