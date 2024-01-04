import pandas as pd
import math
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
class MercatorBox:
    north: float
    east: float
    south: float
    west: float

    def __str__(self):
        return f'mercatorbox-{self.north}-{self.east}-{self.south}-{self.west}'

c =10018754.17139462


def merc_to_latlong(mx: float, my: float):
    #long = (mx / c) * 90.0
    #lat = 180 / math.pi * (2 * math.atan( math.exp( ((my / c) * 90.0) * math.pi / 180.0)) - math.pi / 2.0)
    s =2 * math.pi * 6378137 / 2. 

    long = (mx / s) * 180.0
    lat = (my / s) * 180.0

    lat = 180 / math.pi * (2 * math.atan( math.exp( lat * math.pi / 180.0)) - math.pi / 2.0)
    return lat, long

def latlong_to_merc(lat: float, long: float):
    s =2 * math.pi * 6378137 / 2. 
    #mx = long * c / 90.0
    #my = math.log( math.tan((90 + lat) * math.pi / 360.0 )) / (math.pi / 180.0) * c / 90.0
    mx = long * s / 180.0
    my = math.log( math.tan((90 + lat) * math.pi / 360.0 )) / (math.pi / 180.0)

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


@dataclass
class Map:
    mercator_box: MercatorBox
    image: Image.Image
