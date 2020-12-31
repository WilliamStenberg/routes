from typing import List
from glob import glob
import pandas as pd
from mongoengine import *  # noqa
import matplotlib.pyplot as plt  # Image saving
from uuid import uuid4

from parser import parse_file
from utils import DATAPATH, IMAGEPATH
from utils import file_name_validator, nonnegative_number_validator
from maps import padded_coordinate_bounding_box, fetch_osv_image
connect('rundb', host='127.0.0.1', port=27017)


class Map(Document):
    """
    Map of terrain where routes are run containing image and GPS coordinates

    Init fetches an image from OpenStreetView and stores it along with
    bounding box coordinates.
    """
    title = StringField(max_length=256, required=False)
    bounding_box = PolygonField(auto_index=True, unique=True)
    image_mercator_extent_dict = DictField(unique=True)
    image_path = StringField(unique=True, validation=file_name_validator)
    image_width = FloatField()
    image_height = FloatField()


class Route(Document):
    """
    Route representation with string metadata and aggregated key numbers,
    reference to file and indexed by a lat/long point.
    """
    title = StringField(max_length=256, required=True)
    file_name = StringField(max_length=256, required=True,
                            validation=file_name_validator)
    distance = FloatField(required=True,
                          validation=nonnegative_number_validator)
    location = PointField(auto_index=False)
    datetime = DateTimeField()
    map_ref = ReferenceField(Map)

    meta = {
        'indexes': [[('location', '2dsphere'), ('datetime', 1)]]
    }


def get_map(df: pd.DataFrame) -> Map:
    """
    Fetches a Map object using dataframe coordinate bounding box,
    creating it if nonexistent.
    """
    coord_rect = padded_coordinate_bounding_box(df)
    polygon = coord_rect.bounding_box()
    polygon.append(polygon[0])  # Close the polygon with starting point
    found_maps = Map.objects(
        bounding_box__geo_intersects=polygon)
    if not found_maps:
        image, extent = fetch_osv_image(coord_rect)
        file_path = IMAGEPATH + str(uuid4()) + '.png'
        plt.imsave(file_path, image)
        route_map = Map(
            bounding_box=[polygon],
            image_mercator_extent_dict=extent.__dict__,
            image_path=file_path,
            image_width=image.shape[1],
            image_height=image.shape[0])
        route_map.save()
    else:
        route_map = found_maps[0]
    return route_map


def make_route(file_name: str, df: pd.DataFrame) -> Route:
    """
    Parse file into dataframe and then that into a Route object.
    Constructs the and returns it.
    """
    distance = df['distance'].iloc[len(df) - 1]
    first_non_nan = df['position_lat'].first_valid_index()
    first_row = df.iloc[first_non_nan]
    location = [first_row['position_lat'], first_row['position_long']]
    datetime = first_row['timestamp']
    title = file_name.replace(DATAPATH, '').rstrip('.fit')
    route_map = get_map(df)
    route = Route(title=title,
                  file_name=file_name,
                  distance=distance,
                  location=location,
                  datetime=datetime,
                  map_ref=route_map)
    route.save()
    return route


def refresh_db() -> List[Route]:
    """
    Convenience function to clear DB and replace with new parsings
    """
    routes = []
    dfs = []
    Route.objects().delete()
    Map.objects.delete()
    for file_name in glob(DATAPATH + '*.fit'):
        df = parse_file(file_name)
        dfs.append(df)
        route = make_route(file_name, df)
        routes.append(route)
    return routes, dfs
