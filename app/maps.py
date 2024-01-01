from typing import List, Tuple, Dict
from uuid import uuid4
import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import db as db
import model as model
import utils as utils



def get_map(sess, ts: model.Timeseries) -> db.Map:
    """
    Fetches a Map object using dataframe coordinate bounding box,
    creating it if nonexistent.
    """
    coord_rect = ts.padded_rect()
    found_map = db.smallest_map_enclosing(sess, coord_rect)
    if not found_map:
        image, extent = fetch_osv_image(coord_rect)
        file_path = utils.IMAGEPATH + str(uuid4()) + '.png'
        plt.imsave(file_path, image)
        route_map = db.Map(
            padded_route_bounding_box = db.PaddedRouteBoundingBox(coord_rect),
            mercator_bounding_box = extent,
            image_path=file_path,
            image_width=image.shape[1],
            image_height=image.shape[0],
        )
    else:
        route_map = found_map
    return route_map



def fetch_osv_image(box: model.BoundingBox
                    ) -> Tuple[np.ndarray, db.MercatorBoundingBox]:
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
    return im, db.MercatorBoundingBox(west=min_x, south=min_y, east=max_x, north=max_y)


def transform_geodata(df: pd.DataFrame, image_shape: Tuple[int, int],
                      mercator_box: db.MercatorBoundingBox) -> Tuple[List, List]:
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
