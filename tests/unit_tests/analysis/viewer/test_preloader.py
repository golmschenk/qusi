import os
from asyncio import Task
from collections import deque
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pandas as pd
import pytest

import ramjet.analysis.viewer.preloader as module
from ramjet.analysis.viewer.preloader import Preloader


@pytest.mark.skipif(
    "TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
    reason="Travis CI does not work well with pytest asyncio yet.",
)
class TestPreloader:
    @pytest.mark.asyncio
    async def test_loading_view_entity_at_index_as_current_passes_correct_row(self):
        preloader = Preloader()
        preloader.identifier_data_frame = pd.DataFrame(
            {"light_curve_path": ["a.fits", "b.fits", "c.fits"]}
        )
        stub_light_curve = Mock()
        mock_load_light_curve_from_path = Mock(return_value=stub_light_curve)
        preloader.load_light_curve_from_identifier = mock_load_light_curve_from_path
        index = 1
        preloader.reset_deques = AsyncMock()

        with patch.object(
            module.ViewEntity, "from_identifier_data_frame_row"
        ) as mock_view_entity_factory:
            await preloader.load_view_entity_at_index_as_current(index)

            assert mock_view_entity_factory.call_args.args[0].equals(
                preloader.identifier_data_frame.iloc[index]
            )

    @pytest.mark.asyncio
    async def test_loading_next_view_entities_loads_starting_from_last_in_next_deque(
        self,
    ):
        preloader = Preloader()
        preloader.identifier_data_frame = pd.DataFrame(
            {"light_curve_path": ["a.fits", "b.fits", "c.fits", "d.fits"]}
        )
        stub_view_entity_dictionary = {
            "a.fits": Mock(index=0),
            "b.fits": Mock(index=1),
            "c.fits": Mock(index=2),
            "d.fits": Mock(index=3),
        }

        def load_view_entity_side_effect(identifier_row):
            return stub_view_entity_dictionary[identifier_row["light_curve_path"]]

        preloader.next_view_entity_deque = deque([Mock(index=1)])

        with patch.object(
            module.ViewEntity, "from_identifier_data_frame_row"
        ) as stub_view_entity_factory:
            stub_view_entity_factory.side_effect = load_view_entity_side_effect
            await preloader.load_next_view_entities()

        assert len(preloader.next_view_entity_deque) == 3
        assert preloader.next_view_entity_deque[1].index == 2
        assert preloader.next_view_entity_deque[2].index == 3

    @pytest.mark.asyncio
    async def test_loading_next_view_entities_loads_from_current_when_next_deque_is_empty(
        self,
    ):
        preloader = Preloader()
        preloader.identifier_data_frame = pd.DataFrame(
            {"light_curve_path": ["a.fits", "b.fits", "c.fits", "d.fits"]}
        )
        stub_view_entity_dictionary = {
            "a.fits": Mock(index=0),
            "b.fits": Mock(index=1),
            "c.fits": Mock(index=2),
            "d.fits": Mock(index=3),
        }

        def load_view_entity_side_effect(identifier_row):
            return stub_view_entity_dictionary[identifier_row["light_curve_path"]]

        preloader.current_view_entity = Mock(index=1)

        with patch.object(
            module.ViewEntity, "from_identifier_data_frame_row"
        ) as stub_view_entity_factory:
            stub_view_entity_factory.side_effect = load_view_entity_side_effect
            await preloader.load_next_view_entities()

        assert len(preloader.next_view_entity_deque) == 2
        assert preloader.next_view_entity_deque[0].index == 2
        assert preloader.next_view_entity_deque[1].index == 3

    @pytest.mark.asyncio
    async def test_loading_previous_view_entities_loads_from_first_in_previous_deque(
        self,
    ):
        preloader = Preloader()
        preloader.identifier_data_frame = pd.DataFrame(
            {"light_curve_path": ["a.fits", "b.fits", "c.fits", "d.fits"]}
        )
        stub_view_entity_dictionary = {
            "a.fits": Mock(index=0),
            "b.fits": Mock(index=1),
            "c.fits": Mock(index=2),
            "d.fits": Mock(index=3),
        }

        def load_view_entity_side_effect(identifier_row):
            return stub_view_entity_dictionary[identifier_row["light_curve_path"]]

        preloader.previous_view_entity_deque = deque([Mock(index=2)])

        with patch.object(
            module.ViewEntity, "from_identifier_data_frame_row"
        ) as stub_view_entity_factory:
            stub_view_entity_factory.side_effect = load_view_entity_side_effect
            await preloader.load_previous_view_entities()

        assert len(preloader.previous_view_entity_deque) == 3
        assert preloader.previous_view_entity_deque[0].index == 0
        assert preloader.previous_view_entity_deque[1].index == 1

    @pytest.mark.asyncio
    async def test_loading_previous_light_curves_loads_from_current_when_previous_deque_is_empty(
        self,
    ):
        preloader = Preloader()
        preloader.identifier_data_frame = pd.DataFrame(
            {"light_curve_path": ["a.fits", "b.fits", "c.fits", "d.fits"]}
        )
        stub_view_entity_dictionary = {
            "a.fits": Mock(index=0),
            "b.fits": Mock(index=1),
            "c.fits": Mock(index=2),
            "d.fits": Mock(index=3),
        }

        def load_view_entity_side_effect(identifier_row):
            return stub_view_entity_dictionary[identifier_row["light_curve_path"]]

        preloader.current_view_entity = Mock(index=2)

        with patch.object(
            module.ViewEntity, "from_identifier_data_frame_row"
        ) as stub_view_entity_factory:
            stub_view_entity_factory.side_effect = load_view_entity_side_effect
            await preloader.load_previous_view_entities()

        assert len(preloader.previous_view_entity_deque) == 2
        assert preloader.previous_view_entity_deque[0].index == 0
        assert preloader.previous_view_entity_deque[1].index == 1

    @pytest.mark.asyncio
    async def test_incrementing_preloader_shifts_light_curve_deques(self):
        preloader = Preloader()
        preloader.current_view_entity = Mock(index=1)
        preloader.previous_view_entity_deque = deque([Mock(index=0)])
        preloader.next_view_entity_deque = deque([Mock(index=2)])
        preloader.refresh_surrounding_light_curve_loading = AsyncMock()

        new_current = await preloader.increment()

        assert new_current.index == 2
        assert len(preloader.previous_view_entity_deque) == 2
        assert preloader.previous_view_entity_deque[-1].index == 1
        assert len(preloader.next_view_entity_deque) == 0

    @pytest.mark.asyncio
    async def test_decrementing_preloader_shifts_light_curve_deques(self):
        preloader = Preloader()
        preloader.current_view_entity = Mock(index=1)
        preloader.previous_view_entity_deque = deque([Mock(index=0)])
        preloader.next_view_entity_deque = deque([Mock(index=2)])
        preloader.refresh_surrounding_light_curve_loading = AsyncMock()

        new_current = await preloader.decrement()

        assert new_current.index == 0
        assert len(preloader.next_view_entity_deque) == 2
        assert preloader.next_view_entity_deque[0].index == 1
        assert len(preloader.previous_view_entity_deque) == 0

    @pytest.mark.asyncio
    async def test_incrementing_calls_refresh_of_surrounding_light_curve_loading(self):
        preloader = Preloader()
        preloader.next_view_entity_deque = MagicMock()
        preloader.previous_view_entity_deque = MagicMock()
        preloader.current_view_entity = Mock()
        mock_refresh_surrounding_light_curve_loading = AsyncMock()
        preloader.refresh_surrounding_light_curve_loading = (
            mock_refresh_surrounding_light_curve_loading
        )

        _ = await preloader.increment()

        assert mock_refresh_surrounding_light_curve_loading.called

    @pytest.mark.asyncio
    async def test_decrementing_calls_refresh_of_surrounding_light_curve_loading(self):
        preloader = Preloader()
        preloader.next_view_entity_deque = MagicMock()
        preloader.previous_view_entity_deque = MagicMock()
        preloader.current_view_entity = Mock()
        mock_refresh_surrounding_light_curve_loading = AsyncMock()
        preloader.refresh_surrounding_light_curve_loading = (
            mock_refresh_surrounding_light_curve_loading
        )

        _ = await preloader.decrement()

        assert mock_refresh_surrounding_light_curve_loading.called

    @pytest.mark.asyncio
    async def test_cancel_loading_cancels_existing_loading_task(self):
        preloader = Preloader()
        mock_coroutine_function = AsyncMock()
        mock_old_running_loading_task = Task(mock_coroutine_function())
        preloader.running_loading_task = mock_old_running_loading_task

        await preloader.cancel_loading_task()

        assert mock_old_running_loading_task.cancelled

    @pytest.mark.asyncio
    async def test_cancel_loading_does_not_error_with_no_existing_loading_task(self):
        preloader = Preloader()
        preloader.running_loading_task = None

        await preloader.cancel_loading_task()

        assert True  # Implicit pass if an error did not occur.

    @pytest.mark.asyncio
    async def test_refresh_surrounding_light_curve_loading_creates_loading_task(self):
        preloader = Preloader()
        stub_coroutine = Mock()
        mock_load_surrounding_light_curves = Mock(return_value=stub_coroutine)
        preloader.load_surrounding_view_entities = mock_load_surrounding_light_curves

        with patch.object(module.asyncio, "create_task") as mock_create_task:
            _ = await preloader.refresh_surrounding_light_curve_loading()

            assert mock_create_task.call_args[0][0] is stub_coroutine

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_factory_from_csv_path_loads_view_entities(self):
        identifier_data_frame = pd.DataFrame(
            {"light_curve_path": ["a.fits", "b.fits", "c.fits", "d.fits"]}
        )
        stub_view_entity_dictionary = {
            "a.fits": Mock(index=0),
            "b.fits": Mock(index=1),
            "c.fits": Mock(index=2),
            "d.fits": Mock(index=3),
        }

        def load_view_entity_side_effect(identifier_row):
            return stub_view_entity_dictionary[identifier_row["light_curve_path"]]

        with patch.object(
            module.ViewEntity, "from_identifier_data_frame_row"
        ) as stub_view_entity_factory, patch.object(
            module.pd, "read_csv"
        ) as stub_read_csv:
            stub_read_csv.return_value = identifier_data_frame
            stub_view_entity_factory.side_effect = load_view_entity_side_effect
            preloader = await Preloader.from_csv_path(csv_path=Path(), starting_index=1)
            await preloader.running_loading_task

        assert preloader.current_view_entity == stub_view_entity_dictionary["b.fits"]
        assert len(preloader.previous_view_entity_deque) == 1
        assert len(preloader.next_view_entity_deque) == 2
