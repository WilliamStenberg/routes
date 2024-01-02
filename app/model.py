from typing import List
import pandas as pd
from dataclasses import dataclass
from PIL import Image
class Timeseries:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def bounding_box(self) -> "BoundingBox":
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

    def padded_rect(self) -> "BoundingBox":
        lat_diff = (self.north-self.south) * 0.05
        long_diff = (self.east-self.west) * 0.05
        return BoundingBox(
            north=self.north+lat_diff,
            east=self.east+long_diff,
            south=self.south-lat_diff,
            west=self.west-long_diff)

    def contains(self, other: "BoundingBox") -> bool:
        return self.north >= other.north and self.east >= other.east and self.south <= other.south and self.west <= other.west


@dataclass
class Map:
    mercator_box: BoundingBox
    image: Image.Image
