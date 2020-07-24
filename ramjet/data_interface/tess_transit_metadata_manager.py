"""
Code for managing the TESS transit metadata.
"""
import pandas as pd
from enum import Enum
from peewee import IntegerField, CharField

from ramjet.data_interface.metadatabase import MetadatabaseModel, metadatabase
from ramjet.data_interface.tess_toi_data_interface import TessToiDataInterface, ToiColumns


class Disposition(Enum):
    """
    An enum to represent the possible dispositions of a target.
    """
    CONFIRMED = 'Confirmed'
    CANDIDATE = 'Candidate'
    FALSE_POSITIVE = 'False positive'


class TessTransitMetadata(MetadatabaseModel):
    """
    A model for the TESS target transit metadatabase table.
    """
    tic_id = IntegerField(index=True, unique=True)
    disposition = CharField(index=True, choices=Disposition)

    class Meta:
        """Schema meta data for the model."""
        indexes = (
            (('disposition', 'tic_id'), False),
        )


class TessTransitMetadataManager:
    """
    A class for managing the TESS transit metadata.
    """
    @staticmethod
    def build_table():
        """
        Builds the TESS transit metadata table.
        """
        print('Building TESS transit metadata table...')
        tess_toi_data_interface = TessToiDataInterface()
        toi_dispositions = tess_toi_data_interface.toi_dispositions
        ctoi_dispositions = tess_toi_data_interface.ctoi_dispositions
        toi_filtered_dispositions = toi_dispositions.filter([ToiColumns.tic_id.value, ToiColumns.disposition.value])
        ctoi_filtered_dispositions = ctoi_dispositions.filter([ToiColumns.tic_id.value, ToiColumns.disposition.value])
        all_dispositions = pd.concat([toi_filtered_dispositions, ctoi_filtered_dispositions], ignore_index=True)
        target_grouped_dispositions = all_dispositions.groupby(ToiColumns.tic_id.value)[ToiColumns.disposition.value
                                                                                        ].apply(set)
        row_count = 0
        metadatabase.drop_tables([TessTransitMetadata])
        metadatabase.create_tables([TessTransitMetadata])
        with metadatabase.atomic():
            for tic_id, disposition_set in target_grouped_dispositions.iteritems():
                # As a target can have multiple dispositions, use the most forgiving available disposition.
                if 'KP' in disposition_set or 'CP' in disposition_set:
                    database_disposition = Disposition.CONFIRMED.value
                elif 'PC' in disposition_set or '' in disposition_set:
                    database_disposition = Disposition.CANDIDATE.value
                elif 'FP' in disposition_set:
                    database_disposition = Disposition.FALSE_POSITIVE.value
                else:
                    raise ValueError(f'{disposition_set} does not contain a known disposition.')
                row = TessTransitMetadata(tic_id=tic_id, disposition=database_disposition)
                row.save()
                row_count += 1
        print(f'Table built. {row_count} rows added.')


if __name__ == '__main__':
    tess_transit_metadata_manager = TessTransitMetadataManager()
    tess_transit_metadata_manager.build_table()

