from typing import List
from glob import glob
import pandas as pd
from mongoengine import *  # noqa

from parser import parse_file
from utils import DATAPATH, file_name_validator, nonnegative_number_validator
connect('rundb', host='127.0.0.1', port=27017)


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

    meta = {
        'indexes': [[('location', '2dsphere'), ('datetime', 1)]]
    }


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
    route = Route(title=file_name.replace(DATAPATH, ''),
                  file_name=file_name,
                  distance=distance,
                  location=location,
                  datetime=datetime)
    route.save()
    return route


def refresh_db() -> List[Route]:
    """
    Convenience function to clear DB and replace with new parsings
    """
    routes = []
    dfs = []
    Route.objects().delete()
    for file_name in glob(DATAPATH + '*.fit'):
        df = parse_file(file_name)
        dfs.append(df)
        route = make_route(file_name, df)
        routes.append(route)
    return routes, dfs
