from typing import List
import os
from glob import glob
import pandas as pd
from sqlalchemy import delete

import db as db
import model as model
import parser as parser
import utils as utils

def make_route(sess, file_name: str, df: pd.DataFrame) -> db.Route:
    """
    Parse file into dataframe and then that into a Route object.
    Constructs the and returns it.
    """
    distance = df['distance'].iloc[len(df) - 1]
    first_non_nan = df['position_lat'].first_valid_index()
    first_row = df.iloc[first_non_nan]
    location = [first_row['position_lat'], first_row['position_long']]
    datetime = first_row['timestamp']
    title = file_name.replace(utils.DATAPATH, '').rstrip('.fit')
    route_map, _ = db.persistent_map(sess, model.Timeseries(df))
    sess.add(route_map)
    route = db.Route(title=title,
                  file_name=file_name,
                  distance=distance,
                  start_lat=location[0],
                  start_long=location[1],
                  created_at=datetime,
                  map=route_map)
    sess.add(route)
    return route


def refresh_db() -> List[db.Route]:
    return sync(clear=True)

def clear_maps():
    with db.sess() as sess:
        sess.execute(delete(db.Map))
        sess.commit()
        for file_name in glob(utils.IMAGEPATH + '*.png'):
            os.remove(file_name)

def sync(clear: bool = False) -> List[db.Route]:
    with db.sess() as sess:
        if clear:
            sess.execute(delete(db.Route))
        routes = []
        dfs = []
        existing = [r.file_name for r in db.routes(sess)]
        for file_name in glob(utils.DATAPATH + '*.fit'):
            if file_name not in existing:
                df = parser.parse_file(file_name)
                dfs.append(df)
                route = make_route(sess, file_name, df)
                routes.append(route)
        sess.commit()
        return routes
