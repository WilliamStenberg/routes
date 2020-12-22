from typing import List
import pandas as pd
import numpy as np
import fitdecode as fd
from scipy.ndimage import gaussian_filter1d


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
    def recompute_speed(df: pd.DataFrame) -> None:
        """
        Set 'speed' column of dataframe using numpy differentiation.
        Since measured speed can be flaky, we compute it from distance
        and modify the given dataframe.
        """
        assert 'distance' in df.columns
        df['speed'] = np.diff(df['distance'], prepend=0) * 1000
        df['speed'] = gaussian_filter1d(df['speed'], 15)

    @property
    def columns(self):
        return list(self._columns)

    @columns.setter
    def columns(self, columns: List[str]):
        self._columns = self._columns.union(columns)

    def get_dataframe(self) -> pd.DataFrame:
        """ Produce Pandas dataframe and do column modifications here """
        df = pd.DataFrame(self.row_dicts)
        RouteProcessor.recompute_speed(df)
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
                    fit.processor.columns = [
                        field_def.name for field_def in frame.all_field_defs
                    ]
            elif isinstance(frame, fd.FitDataMessage):
                pass
        return fit.processor.get_dataframe()
