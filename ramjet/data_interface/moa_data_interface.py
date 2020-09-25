"""
Code for interacting with MOA light curve files and metadata.
"""
import pandas as pd
from collections import defaultdict
from typing import List, Dict

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

    @staticmethod
    def get_tag_for_path_from_data_frame(path: Path, events_data_frame: pd.DataFrame) -> str:
        """
        Gets the event tag of a light curve from the events data frame.

        :param path: The path of the light curve whose event tag should be retrieved.
        :param events_data_frame: Takahiro Sumi's 9-year events data frame.
        :return: The string of the tag of the event. None if no tag exists.
        """
        file_name = path.name
        file_name_without_extension = file_name.split('.')[0]
        moa_identifier = file_name_without_extension.split('_')[-1]  # Remove duplicate identifier string.
        field, clr, chip_string, subfield_string, id_string = moa_identifier.split('-')
        chip, subfield, id_ = int(chip_string), int(subfield_string), int(id_string)
        try:
            row = events_data_frame.loc[(field, clr, chip, subfield, id_)]
            tag = row['tag']
            return tag
        except KeyError:
            return 'no_tag'

    def group_paths_by_tag_in_events_data_frame(self, paths: List[Path], events_data_frame: pd.DataFrame
                                                ) -> Dict[str, List[Path]]:
        """
        Groups paths into a dictionary based on their tags.

        :param paths: The paths to group.
        :param events_data_frame: The events data frame to look for a tag in.
        :return:
        """
        tag_path_list_dictionary = defaultdict(list)
        for path in paths:
            tag = self.get_tag_for_path_from_data_frame(path, events_data_frame)
            tag_path_list_dictionary[tag].append(path)
        return tag_path_list_dictionary
