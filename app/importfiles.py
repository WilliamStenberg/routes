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
    first_row = df.iloc[df['position_lat'].first_valid_index()]
    location = [first_row['position_lat'], first_row['position_long']]
    datetime = first_row['timestamp']
    title = file_name.replace(utils.DATAPATH, '').rstrip('.fit')
    route_map, _ = db.ensure_persistent_map(sess, model.Timeseries(df).bounding_box())
    sess.add(route_map)
    last_row = df.iloc[df['position_lat'].last_valid_index()]
    avg_pace = model.pace_to_str(model.calculate_pace(last_row['distance'], last_row['timestamp']))
    avg_heartrate = None
    print(f'Is heart rate in {df.columns}???\n?\n?\n?')
    if 'heart_rate' in df.columns and (rs := df['heart_rate'].dropna()).size > 0:
        avg_heartrate = rs.mean()
    route = db.Route(title=title,
                  file_name=file_name,
                  distance=distance,
                  avg_pace=avg_pace,
                  avg_heartrate=avg_heartrate,
                  start_lat=location[0],
                  start_long=location[1],
                  created_at=datetime,
                  map=route_map)
    sess.add(route)
    return route


def refresh_db() -> List[db.Route]:
    return sync(clear=True)

def clear_maps():
    db.setup()
    with db.sess() as sess:
        sess.execute(delete(db.Map))
        sess.execute(delete(db.MercatorBoundingBox))
        sess.execute(delete(db.PaddedRouteBoundingBox))
        sess.commit()
        for file_name in glob(utils.IMAGEPATH + '*.png'):
            os.remove(file_name)

def sync(clear: bool = False) -> List[db.Route]:
    db.setup()
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
