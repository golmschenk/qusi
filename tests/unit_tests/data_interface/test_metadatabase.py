from uuid import UUID

import pytest

from ramjet.data_interface.metadatabase import (
    dataset_split_from_uuid,
    metadatabase_uuid,
)


class TestMetadatabase:
    @pytest.mark.integration
    def test_metadatabase_uuid_is_repeatable(self):
        uuid = metadatabase_uuid("FakeDatabaseTable TIC 1234567 sector 1")
        assert uuid == UUID("80f551f9-b8f9-5146-8059-65bc116883c4")

    def test_dataset_split_is_repeatable_from_uuid(self):
        dataset_split = dataset_split_from_uuid(
            UUID("80f551f9-b8f9-5146-8059-65bc116883c4")
        )
        assert dataset_split == 6
