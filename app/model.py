from typing import List, Tuple
import pandas as pd
import math
from dataclasses import dataclass
from datetime import timedelta
from PIL import Image


s = 2 * math.pi * 6378137 / 2.0


class Timeseries:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def bounding_box(self) -> 'BoundingBox':
        """
        Extract dataframe GPS west-south-east-north route mins/max,
        add margins and return the four values
        """
        west = self.df['position_long'].min()
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

    def padded_rect(self) -> 'BoundingBox':
        lat_diff = (self.north - self.south) * 0.05
        long_diff = (self.east - self.west) * 0.05
        return BoundingBox(
            north=self.north + lat_diff,
            east=self.east + long_diff,
            south=self.south - lat_diff,
            west=self.west - long_diff)

    def contains(self, other: 'BoundingBox') -> bool:
        ne = self.north >= other.north and self.east >= other.east
        sw = self.south <= other.south and self.west <= other.west
        return ne and sw

    def contains_point(self, lat: float, long: float) -> bool:
        ne = self.north >= lat and self.east >= long
        sw = self.south <= lat and self.west <= long
        return ne and sw


@dataclass
class MercatorBox:
    north: float
    east: float
    south: float
    west: float

    def __str__(self):
        return f'mercatorbox-{self.north}-{self.east}-{self.south}-{self.west}'

    def contains_point(self, mx: float, my: float) -> bool:
        return self.north >= my and self.east >= mx and self.south <= my and self.west <= mx


def merc_to_latlong(mx: float, my: float):
    long = (mx / s) * 180.0
    lat = (my / s) * 180.0
    lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180.0)) - math.pi / 2.0)
    return lat, long


def latlong_to_merc(lat: float, long: float):
    mx = long * s / 180.0
    my = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    my = my * s / 180.0
    return mx, my


def merc_box_to_latlong(mercbox: MercatorBox) -> BoundingBox:
    (north, west) = merc_to_latlong(mercbox.west, mercbox.north)
    (south, east) = merc_to_latlong(mercbox.east, mercbox.south)

    return BoundingBox(
        north=north,
        east=east,
        south=south,
        west=west)


def latlong_box_to_merc(box: BoundingBox) -> MercatorBox:
    (west, north) = latlong_to_merc(box.north, box.west)
    (east, south) = latlong_to_merc(box.south, box.east)

    return MercatorBox(
        north=north,
        east=east,
        south=south,
        west=west)


def box_around_boxes(boxes: List[BoundingBox]) -> BoundingBox:
    return BoundingBox(
        north=max([b.north for b in boxes]),
        east=max([b.east for b in boxes]),
        south=min([b.south for b in boxes]),
        west=min([b.west for b in boxes]))


def box_around_latlong_points(points: List[Tuple[float, float]]) -> BoundingBox:
    return BoundingBox(
        north=max([b[0] for b in points]),
        east=max([b[1] for b in points]),
        south=min([b[0] for b in points]),
        west=min([b[1] for b in points]))


@dataclass
class Map:
    mercator_box: MercatorBox
    image: Image.Image


def calculate_pace(delta_dist: float, delta_time: timedelta) -> timedelta:
    try:
        speed = abs(delta_dist) / (delta_time).seconds
        # Convert to seconds / km as timedelta
        pace = timedelta(seconds=1 / speed) if speed > 0 else timedelta(seconds=0)
        return pace
    except Exception:
        return timedelta(seconds=0)


def moment_pace(delta_dist: float) -> timedelta:
    if delta_dist < 1 or delta_dist != delta_dist:
        return timedelta(seconds=0)
    return timedelta(seconds=1000 / delta_dist)


def pace_to_str(t: timedelta) -> str:
    if t.seconds < 60:
        return f'{t.seconds:02}s'
    elif t.seconds % 60 == 0:
        return f'{t.seconds // 60}min'
    return f'{t.seconds//60}min, {(t.seconds % 60):02}s'
