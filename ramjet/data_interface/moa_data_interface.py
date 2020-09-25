"""
Code for interacting with MOA light curve files and metadata.
"""
from typing import Union

import pandas as pd

from pathlib import Path


class MoaDataInterface:
    """
    A class for interacting with MOA light curve files and metadata.
    """
    @staticmethod
    def read_takahiro_sumi_nine_year_events_table_as_data_frame(path: Path) -> pd.DataFrame:
        """
        Reads Takahiro Sumi's 9-year events table as a Pandas data frame.

        :param path: The path to the events table file.
        :return: The data frame.
        """
        column_names = ['field', 'clr', 'chip', 'subfield', 'id', 'tag', 'x', 'y', '2006_2007_tag',
                        '2006_2007_separation', '2006_2007_id', '2006_2007_x', '2006_2007_y', 'alert_tag',
                        'alert_separation', 'alert_name', 'alert_x', 'alert_y']
        widths = [4, 2, 3, 2, 7, 3, 10, 10, 3, 6, 13, 10, 10, 3, 6, 13, 10, 10]
        data_frame = pd.read_fwf(path, comment='#', skiprows=23, names=column_names, widths=widths)
        data_frame = data_frame.set_index(['field', 'clr', 'chip', 'subfield', 'id'], drop=False)
        data_frame = data_frame.sort_index()
        return data_frame

    def get_tag_for_path_from_data_frame(self, path: Path, data_frame: pd.DataFrame) -> Union[str, None]:
        """
        Gets the event tag of a light curve from the events data frame.

        :param path: The path of the light curve whose event tag should be retrieved.
        :param data_frame: Takahiro Sumi's 9-year events data frame.
        :return: The string of the tag of the event. None if no tag exists.
        """
        file_name = path.name
        file_name_without_extension = file_name.split('.')[0]
        moa_identifier = file_name_without_extension.split('_')[-1]  # Remove duplicate identifier string.
        field, clr, chip_string, subfield_string, id_string = moa_identifier.split('-')
        chip, subfield, id = int(chip_string), int(subfield_string), int(id_string)
        try:
            row = data_frame.loc[(field, clr, chip, subfield, id)]
            tag = row['tag']
            return tag
        except KeyError:
            return None
