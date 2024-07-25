"""
Code for interacting with MOA light curve files and metadata.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import ClassVar

import pandas as pd
import requests
from bs4 import BeautifulSoup


class MoaDataInterface2M:
    """
    A class for interacting with MOA light curve files and metadata.
    """

    survey_tag_to_path_list_dictionary_: dict[str, list[Path]] | None = None
    no_tag_string = "no_tag"
    all_survey_tags: ClassVar[list[str]] = ["c", "cf", "cp", "cw", "cs", "cb", "v", "n", "nr", "m", "j", no_tag_string]

    @property
    def survey_tag_to_path_list_dictionary(self) -> dict[str, list[Path]]:
        """
        Property allowing the survey tag to path list dictionary to only be loaded once.

        :return: The survey tag to path list dictionary
        """
        if self.survey_tag_to_path_list_dictionary_ is None:
            takahiro_sumi_nine_year_events_data_frame = self.read_takahiro_sumi_nine_year_events_table_as_data_frame(
                Path("/local/data/fugu3/sishitan/qusi_project/qusi/data/moa_microlensing_550k/candlist_2023Oct12.txt")
            )
            self.survey_tag_to_path_list_dictionary_ = self.group_paths_by_tag_in_events_data_frame(
                list(Path("/local/data/fugu3/sishitan/qusi_project/merida/data/microlensing_2M").glob("**/*.feather")), takahiro_sumi_nine_year_events_data_frame
            )
        return self.survey_tag_to_path_list_dictionary_


    @staticmethod
    def read_takahiro_sumi_nine_year_events_table_as_data_frame(path: Path) -> pd.DataFrame:
        """
        Reads Takahiro Sumi's 9-year events table as a Pandas data frame.

        :param path: The path to the events table file.
        :return: The data frame.
        """
        named_column_names = ["field", "clr", "chip", "subfield", "id", "tag", "x", "y"]
        # The number of columns in the file are inconsistent, so here we add extra unnamed columns to match the
        # largest number of columns in any row.
        largest_column_count = 33
        unnamed_column_names = [f"unnamed{index}" for index in range(largest_column_count - len(named_column_names))]
        column_names = named_column_names + unnamed_column_names
        data_frame = pd.read_csv(
            path, comment="#", names=column_names, delim_whitespace=True, skipinitialspace=True, skiprows=23
        )
        data_frame = data_frame.set_index(["field", "clr", "chip", "subfield", "id"], drop=False)
        data_frame = data_frame.sort_index()
        return data_frame

    def get_tag_for_path_from_data_frame(self, path: Path, events_data_frame: pd.DataFrame) -> str:
        """
        Gets the event tag of a light curve from the events data frame.

        :param path: The path of the light curve whose event tag should be retrieved.
        :param events_data_frame: Takahiro Sumi's 9-year events data frame.
        :return: The string of the tag of the event. None if no tag exists.
        """
        file_name = path.name
        file_name_without_extension = file_name.split(".")[0]
        moa_identifier = file_name_without_extension.split("_")[-1]  # Remove duplicate identifier string.
        field, clr, chip_string, subfield_string, id_string = moa_identifier.split("-")
        chip, subfield, id_ = int(chip_string), int(subfield_string), int(id_string)
        try:
            row = events_data_frame.loc[(field, clr, chip, subfield, id_)]
        except KeyError:
            return self.no_tag_string
        tag = row["tag"]
        return tag

    def group_paths_by_tag_in_events_data_frame(
        self, paths: list[Path], events_data_frame: pd.DataFrame
    ) -> dict[str, list[Path]]:
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
