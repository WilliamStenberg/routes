from typing import List
import pandas as pd
import numpy as np
import fitdecode as fd
from scipy.ndimage import gaussian_filter1d
from timezonefinder import TimezoneFinder
from dateutil import tz
from datetime import timedelta

import model as model


class RouteProcessor(fd.StandardUnitsDataProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._columns = set()
        self.columns = list()
        self.row_dicts = list()

    def process_message_record(self, reader, msg) -> None:
        """ This is automatically called for every datapoint """
        row_dict = {field.name: field.value for field in msg.fields
                    if field.name in self.columns}
        self.row_dicts.append(row_dict)

    @staticmethod
    def recompute_speed(df: pd.DataFrame) -> pd.DataFrame:
        """
        Set 'speed' column of dataframe using numpy differentiation.
        Since measured speed can be flaky, we compute it from distance
        and return modified dataframe.
        """
        assert 'distance' in df.columns
        df['speed'] = np.diff(df['distance'], prepend=0) * 1000
        df['speed'] = gaussian_filter1d(df['speed'], 15)
        return df

    @staticmethod
    def adjust_timestamp_by_timezone(df: pd.DataFrame) -> pd.DataFrame:
        """
        Get timezone from first GPS coord, convert all timestamps
        to that zone and return modified dataframe.
        """
        tf = TimezoneFinder()
        gps_row = df.iloc[df['position_lat'].first_valid_index()]
        latitude = gps_row['position_lat']
        longitude = gps_row['position_long']
        # Zone string name, e.g. 'America/New_York'
        zone_name = tf.timezone_at(lng=longitude, lat=latitude)
        zone = tz.gettz(zone_name)  # In dateutil format
        df['timestamp'] = df['timestamp'].map(lambda t: t.astimezone(zone))
        return df

    @property
    def columns(self):
        return list(self._columns)

    @columns.setter
    def columns(self, columns: List[str]):
        self._columns = self._columns.union(columns)

    def get_dataframe(self) -> pd.DataFrame:
        """ Produce Pandas dataframe and do column modifications here """
        df = pd.DataFrame(self.row_dicts)
        df = RouteProcessor.recompute_speed(df)
        df = RouteProcessor.adjust_timestamp_by_timezone(df)
        return df


def parse_file(file_name: str) -> pd.DataFrame:
    """
    Read FIT file using above processor. Set columns from field definition
    messages with "record" tag.
    """
    with fd.FitReader(file_name, processor=RouteProcessor()) as fit:
        for frame in fit:
            if isinstance(frame, fd.FitDefinitionMessage):
                if frame.name == 'record':
                    fit.processor.columns = [ # type: ignore
                        field_def.name for field_def in frame.all_field_defs
                    ]
            elif isinstance(frame, fd.FitDataMessage):
                pass
        return fit.processor.get_dataframe()  # type: ignore


class SectionPaceInfo:
    """
    Running pace information for a section of a route

    Tied to a dataframe with indices to start and stop, and aggregated
    section distance and duration. Keeps the pace as "minutes per kilometer"
    in timedelta format.
    """
    def __init__(self, distance: float, duration: timedelta,
                 mins_per_km: timedelta, start_index: int, end_index: int):
        self.distance = distance
        self.duration = duration
        self.mins_per_km = mins_per_km
        self.start_index = start_index
        self.end_index = end_index

    def formatted_pace(self) -> str:
        return model.f(self.duration)


def section_pace_infos(df, kilometer_distance_steps: float = 1,
                       include_total: bool = True) -> List[SectionPaceInfo]:
    """
    Compute route sections by given distance step

    Steps through dataframe and computes average speed as distance
    over time and converts to running pace.
    """
    ref_index = 0
    valid_index = df['distance'].first_valid_index()
    ref_time, ref_dist = df.loc[valid_index][['timestamp', 'distance']]
    objects = list()
    while True:
        delta_index = df.loc[ref_index:]['distance'].searchsorted(
            ref_dist + kilometer_distance_steps)
        if delta_index <= 1:
            break
        elif ref_index + delta_index >= len(df):
            delta_index = len(df) - ref_index - 1
        next_index = ref_index + delta_index - 1
        new_time, new_dist = df.loc[next_index][['timestamp', 'distance']]
        # Average speed in km/s
        delta_dist, delta_time = new_dist - ref_dist, new_time - ref_time
        pace = model.calculate_pace(delta_dist, delta_time)
        objects.append(SectionPaceInfo(
            delta_dist, delta_time, pace, ref_index, ref_index + delta_index))
        ref_time = new_time
        ref_dist = new_dist
        ref_index = ref_index + delta_index
    if include_total:
        # One-time recursive call to include average pace for the entire route
        # by giving maximum step distance
        objects += section_pace_infos(df, np.iinfo(np.int64).max,
                                      include_total=False)
    return objects

