from typing import List, Tuple, Dict
import pandas as pd
import geopandas as gpd
import numpy as np
import contextily as ctx


class CoordinateRectangle:
    """
    Four coordinates specifying a rectangle. Should be interpreted as lat/long
    coordinates unless mercator property is explicitly set to True

    Can be accessed by west-south-east-north properties or as
    a list of four coordinates (southwest, ...)
    """
    def __init__(self, west: float, south: float, east: float, north: float,
                 mercator: bool = False):
        self.west = west
        self.south = south
        self.east = east
        self.north = north
        self.mercator = mercator

    def bounding_box(self) -> List[Tuple[float, float]]:
        """
        Pair up coordinates to four corners in a bounding box

        Returns coordinates (in either representation) for
        southwest, southeast, northeast, and northwest corners.
        """
        return [[self.west, self.south], [self.east, self.south],
                [self.east, self.north], [self.west, self.north]]


def padded_coordinate_bounding_box(df: pd.DataFrame) -> CoordinateRectangle:
    """
    Extract dataframe GPS west-south-east-north route mins/max,
    add margins and return the four values
    """
    assert 'position_lat' in df.columns and 'position_long' in df.columns
    lat_diff, long_diff = 0.0005, 0.001  # Add ~50m off each limit
    num_margin = 2
    west = df['position_long'].min() - long_diff * num_margin
    south = df['position_lat'].min() - lat_diff * num_margin
    east = df['position_long'].max() + long_diff * num_margin
    north = df['position_lat'].max() + lat_diff * num_margin
    return CoordinateRectangle(west, south, east, north)


def fetch_osv_image(box: CoordinateRectangle
                    ) -> Tuple[np.ndarray, CoordinateRectangle]:
    """
    Get satellite image from OpenStreetView as Numpy array for given lat/long
    coordinate rectangle (bounding box)

    Includes the image's extent (image corners) as a new coordinate rectangle
    in WebMercator projection.
    """
    assert not box.mercator
    im, extent = ctx.bounds2img(box.west, box.south, box.east, box.north,
                                ll=True,
                                source=ctx.providers.OpenStreetMap.Mapnik)
    # Extent is given in another order
    [min_x, max_x, min_y, max_y] = extent
    return im, CoordinateRectangle(min_x, min_y, max_x, max_y, mercator=True)


def transform_geodata(df: pd.DataFrame, image_shape: Tuple[float, float],
                      extent_dict: Dict) -> Tuple[List, List]:
    """
    Compute x and y lists of route geodata in origin-based reference frame

    Given a dataframe with image shape and mercator bounding box,
    converts lat/long to WebMercator and scales and translates.
    """
    max_x = extent_dict['east']
    min_x = extent_dict['west']
    min_y = extent_dict['south']
    max_y = extent_dict['north']
    scale_factor_x = image_shape[1] / (max_x - min_x)
    scale_factor_y = image_shape[0] / (max_y - min_y)

    raw_gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.position_long, df.position_lat))
    gdf = raw_gdf.set_crs(4327)  # Coordinates are in lat/long WGS format
    gdf = gdf.to_crs('epsg:3857')  # Transform the geometry to WebMercator
    xs = (gdf.geometry.x - min_x) * scale_factor_x
    ys = (gdf.geometry.y - min_y) * scale_factor_y
    return xs, ys
