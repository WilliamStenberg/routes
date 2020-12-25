from typing import List
from glob import glob

from parser import parse_file
from db import Route, make_route
from utils import DATAPATH


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
