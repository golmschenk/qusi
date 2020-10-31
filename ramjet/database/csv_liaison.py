"""
Code for converting components of the database to and from a CSV format for access by those not familiar with SQL.
"""
import pandas as pd
from pathlib import Path

from ramjet.database.base_model import database
from ramjet.database.tess_target import TessTarget
from ramjet.database.tess_planet_disposition import TessPlanetDisposition, Source, Disposition
from ramjet.database.tess_transiter import TessTransiter


class CsvLiaison:
    """
    A class for converting components of the database to and from a CSV format.
    """
    @staticmethod
    def add_vetting_csv_content_to_database(vetting_csv_path: Path) -> None:
        """
        Converts the CSV form of the vetting to database entries.

        :param vetting_csv_path: The path to the CSV file containing the vetting.
        """
        with database.atomic():
            vetting_data_frame = pd.read_csv(vetting_csv_path)
            for index, row in vetting_data_frame.iterrows():
                # Add the target.
                target = TessTarget()
                target.tic_id = row['tic_id']
                existing_target = TessTarget.get_or_none(TessTarget.tic_id == target.tic_id)
                if existing_target is None:
                    target.save()
                else:
                    target = existing_target
                # Add the transiter.
                transiter = TessTransiter()
                transiter.target = target
                transiter.radius__solar_radii = row['transiter_radius__solar_radii']
                transiter.has_known_contamination_ratio = row['has_known_contamination_ratio']
                transiter.transit_epoch__btjd = row['transit_epoch__btjd']
                transiter.transit_period__days = row['transit_period__days']
                transiter.transit_duration__days = row['transit_duration__hours'] / 24
                transiter.transit_relative_depth = row['transit_depth__ppm'] / 1e6
                existing_transiter = TessTransiter.get_or_none(TessTransiter.target == transiter.target)
                if existing_transiter is not None:
                    transiter.id = existing_transiter.id
                transiter.save()
                # Add Greg Olmschenk's disposition.
                if pd.notna(row['greg_disposition']):
                    greg_transit_disposition = TessPlanetDisposition()
                    greg_transit_disposition.transiter = transiter
                    greg_transit_disposition.disposition = Disposition(row['greg_disposition']).value
                    greg_transit_disposition.source = Source.GREG_OLMSCHENK.value
                    if pd.notna(row['greg_notes']):
                        greg_transit_disposition.notes = row['greg_notes']
                    existing_greg_transit_disposition = TessPlanetDisposition.get_or_none(
                        TessPlanetDisposition.transiter == greg_transit_disposition.transiter)
                    if existing_greg_transit_disposition is not None:
                        greg_transit_disposition.id = existing_greg_transit_disposition.id
                    greg_transit_disposition.save()
                # Add the group disposition.
                if pd.notna(row['group_disposition']):
                    group_transit_disposition = TessPlanetDisposition()
                    group_transit_disposition.transiter = transiter
                    group_transit_disposition.disposition = Disposition(row['group_disposition']).value
                    group_transit_disposition.source = Source.GSFC_GROUP.value
                    if pd.notna(row['group_notes']):
                        group_transit_disposition.notes = row['group_notes']
                    existing_group_transit_disposition = TessPlanetDisposition.get_or_none(
                        TessPlanetDisposition.transiter == group_transit_disposition.transiter)
                    if existing_group_transit_disposition is not None:
                        group_transit_disposition.id = existing_group_transit_disposition.id
                    group_transit_disposition.save()


if __name__ == '__main__':
    database.create_tables([TessTarget, TessTransiter, TessPlanetDisposition])
    liaison = CsvLiaison()
    liaison.add_vetting_csv_content_to_database(
        Path('data/Vetting of Ramjet 2020-10-13 candidates - vetting_data_frame.csv'))
