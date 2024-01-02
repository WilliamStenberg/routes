from typing import List, Tuple
import contextily as ctx
import geopandas as gpd
import numpy as np
import pandas as pd

import model as model
import utils as utils
from PIL import Image



def transient_map(box: model.BoundingBox) -> model.Map:
    """
    Fetches a Map object using dataframe coordinate bounding box,
    creating it if nonexistent.
    """
    ndarray, extent = fetch_osv_image(box)
    im = Image.fromarray(ndarray)
    return model.Map(image=im, mercator_box=extent)


def fetch_osv_image(box: model.BoundingBox
                    ) -> Tuple[np.ndarray, model.BoundingBox]:
    """
    Get satellite image from OpenStreetView as Numpy array for given lat/long
    coordinate rectangle (bounding box)

    Includes the image's extent (image corners) as a new coordinate rectangle
    in WebMercator projection.
    """
    osv_source = ctx.providers.OpenStreetMap.Mapnik # type: ignore
    im, extent = ctx.bounds2img(box.west, box.south, box.east, box.north,
                                ll=True,
                                source=osv_source)
    # Extent is given in another order
    [min_x, max_x, min_y, max_y] = extent
    return im, model.BoundingBox(west=min_x, south=min_y, east=max_x, north=max_y)


def transform_geodata(df: pd.DataFrame, image_shape: Tuple[int, int],
                      mercator_box: model.BoundingBox) -> Tuple[List, List]:
    """
    Compute x and y lists of route geodata in origin-based reference frame

    Given a dataframe with image shape and mercator bounding box,
    converts lat/long to WebMercator and scales and translates.
    """
    max_x = mercator_box.east
    min_x = mercator_box.west
    min_y = mercator_box.south
    max_y = mercator_box.north
    scale_factor_x = image_shape[1] / (max_x - min_x)
    scale_factor_y = image_shape[0] / (max_y - min_y)

    raw_gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.position_long, df.position_lat))
    gdf = raw_gdf.set_crs(4327)  # Coordinates are in lat/long WGS format
    gdf = gdf.to_crs('epsg:3857')  # Transform the geometry to WebMercator
    if gdf is not None and gdf.geometry is not None:
        xs = (gdf.geometry.x - min_x) * scale_factor_x
        ys = (gdf.geometry.y - min_y) * scale_factor_y
        return xs, ys
    return [], []
