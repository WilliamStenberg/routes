from typing import List
import pandas as pd
from dataclasses import dataclass
class Timeseries:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def padded_rect(self) -> "BoundingBox":
        """
        Extract dataframe GPS west-south-east-north route mins/max,
        add margins and return the four values
        """
        lat_diff, long_diff = 0.0005, 0.001  # Add ~50m off each limit
        num_margin = 2
        west = self.df['position_long'].min() - long_diff * num_margin
        south = self.df['position_lat'].min() - lat_diff * num_margin
        east = self.df['position_long'].max() + long_diff * num_margin
        north = self.df['position_lat'].max() + lat_diff * num_margin
        return BoundingBox(north, east, south, west)

    def bounding_rect(self) -> "BoundingBox":
        """
        Extract dataframe GPS west-south-east-north route mins/max,
        add margins and return the four values
        """
        west =self.df['position_long'].min()
        south = self.df['position_lat'].min()
        east = self.df['position_long'].max()
        north = self.df['position_lat'].max()
        return BoundingBox(north, east, south, west)  # type: ignore

@dataclass
class BoundingBox:
    north: float
    east: float
    south: float
    west: float

    def __str__(self):
        return f'box-{self.north}-{self.east}-{self.south}-{self.west}'
