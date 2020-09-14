from asyncio import Task
from collections import deque
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

import pytest

import ramjet.analysis.viewer.preloader as module
from ramjet.analysis.viewer.preloader import Preloader, IndexLightCurvePair


class TestPreloader:
    @pytest.mark.asyncio
    async def test_loading_light_curve_at_index_as_current_calls_loading_with_correct_path(self):
        preloader = Preloader()
        preloader.light_curve_identifiers = ['a.fits', 'b.fits', 'c.fits']
        stub_light_curve = Mock()
        mock_load_light_curve_from_path = Mock(return_value=stub_light_curve)
        preloader.load_light_curve_from_identifier = mock_load_light_curve_from_path
        index = 1
        preloader.reset_deques = AsyncMock()

        await preloader.load_light_curve_at_index_as_current(index)

        assert preloader.current_index_light_curve_pair.index == index
        assert preloader.current_index_light_curve_pair.light_curve is stub_light_curve
        assert preloader.load_light_curve_from_identifier.call_args[0][0] == 'b.fits'

    @pytest.mark.asyncio
    async def test_loading_next_light_curves_loads_from_last_in_next_deque(self):
        preloader = Preloader()
        preloader.light_curve_identifiers = ['a.fits', 'b.fits', 'c.fits', 'd.fits']
        stub_light_curves_dictionary = {'a.fits': Mock(), 'b.fits': Mock(), 'c.fits': Mock(), 'd.fits': Mock()}
        def load_light_curve_side_effect(path):
            return stub_light_curves_dictionary[path]
        mock_load_light_curve_from_path = Mock(side_effect=load_light_curve_side_effect)
        preloader.load_light_curve_from_identifier = mock_load_light_curve_from_path
        preloader.next_index_light_curve_pair_deque = deque(
            [IndexLightCurvePair(1, stub_light_curves_dictionary['b.fits'])])

        await preloader.load_next_light_curves()

        assert len(preloader.next_index_light_curve_pair_deque) == 3
        assert preloader.next_index_light_curve_pair_deque[1].index == 2
        assert preloader.next_index_light_curve_pair_deque[1].light_curve == stub_light_curves_dictionary['c.fits']
        assert preloader.next_index_light_curve_pair_deque[2].index == 3
        assert preloader.next_index_light_curve_pair_deque[2].light_curve == stub_light_curves_dictionary['d.fits']

    @pytest.mark.asyncio
    async def test_loading_next_light_curves_loads_from_current_when_next_deque_is_empty(self):
        preloader = Preloader()
        preloader.light_curve_identifiers = ['a.fits', 'b.fits', 'c.fits', 'd.fits']
        stub_light_curves_dictionary = {'a.fits': Mock(), 'b.fits': Mock(), 'c.fits': Mock(), 'd.fits': Mock()}
        def load_light_curve_side_effect(path):
            return stub_light_curves_dictionary[path]
        mock_load_light_curve_from_path = Mock(side_effect=load_light_curve_side_effect)
        preloader.load_light_curve_from_identifier = mock_load_light_curve_from_path
        preloader.current_index_light_curve_pair = IndexLightCurvePair(1, stub_light_curves_dictionary['b.fits'])

        await preloader.load_next_light_curves()

        assert len(preloader.next_index_light_curve_pair_deque) == 2
        assert preloader.next_index_light_curve_pair_deque[0].index == 2
        assert preloader.next_index_light_curve_pair_deque[0].light_curve == stub_light_curves_dictionary['c.fits']
        assert preloader.next_index_light_curve_pair_deque[1].index == 3
        assert preloader.next_index_light_curve_pair_deque[1].light_curve == stub_light_curves_dictionary['d.fits']

    @pytest.mark.asyncio
    async def test_loading_previous_light_curves_loads_from_first_in_previous_deque(self):
        preloader = Preloader()
        preloader.light_curve_identifiers = ['a.fits', 'b.fits', 'c.fits', 'd.fits']
        stub_light_curves_dictionary = {'a.fits': Mock(), 'b.fits': Mock(), 'c.fits': Mock(), 'd.fits': Mock()}
        def load_light_curve_side_effect(path):
            return stub_light_curves_dictionary[path]
        mock_load_light_curve_from_path = Mock(side_effect=load_light_curve_side_effect)
        preloader.load_light_curve_from_identifier = mock_load_light_curve_from_path
        preloader.previous_index_light_curve_pair_deque = deque(
            [IndexLightCurvePair(2, stub_light_curves_dictionary['c.fits'])])

        await preloader.load_previous_light_curves()

        assert len(preloader.previous_index_light_curve_pair_deque) == 3
        assert preloader.previous_index_light_curve_pair_deque[0].index == 0
        assert preloader.previous_index_light_curve_pair_deque[0].light_curve == stub_light_curves_dictionary['a.fits']
        assert preloader.previous_index_light_curve_pair_deque[1].index == 1
        assert preloader.previous_index_light_curve_pair_deque[1].light_curve == stub_light_curves_dictionary['b.fits']


    @pytest.mark.asyncio
    async def test_loading_previous_light_curves_loads_from_first_in_previous_deque(self):
        preloader = Preloader()
        preloader.light_curve_identifiers = ['a.fits', 'b.fits', 'c.fits', 'd.fits']
        stub_light_curves_dictionary = {'a.fits': Mock(), 'b.fits': Mock(), 'c.fits': Mock(), 'd.fits': Mock()}
        def load_light_curve_side_effect(path):
            return stub_light_curves_dictionary[path]
        mock_load_light_curve_from_path = Mock(side_effect=load_light_curve_side_effect)
        preloader.load_light_curve_from_identifier = mock_load_light_curve_from_path
        preloader.previous_index_light_curve_pair_deque = deque(
            [IndexLightCurvePair(2, stub_light_curves_dictionary['c.fits'])])

        await preloader.load_previous_light_curves()

        assert len(preloader.previous_index_light_curve_pair_deque) == 3
        assert preloader.previous_index_light_curve_pair_deque[0].index == 0
        assert preloader.previous_index_light_curve_pair_deque[0].light_curve == stub_light_curves_dictionary['a.fits']
        assert preloader.previous_index_light_curve_pair_deque[1].index == 1
        assert preloader.previous_index_light_curve_pair_deque[1].light_curve == stub_light_curves_dictionary['b.fits']

    @pytest.mark.asyncio
    async def test_loading_previous_light_curves_loads_from_current_when_previous_deque_is_empty(self):
        preloader = Preloader()
        preloader.light_curve_identifiers = ['a.fits', 'b.fits', 'c.fits', 'd.fits']
        stub_light_curves_dictionary = {'a.fits': Mock(), 'b.fits': Mock(), 'c.fits': Mock(), 'd.fits': Mock()}
        def load_light_curve_side_effect(path):
            return stub_light_curves_dictionary[path]
        mock_load_light_curve_from_path = Mock(side_effect=load_light_curve_side_effect)
        preloader.load_light_curve_from_identifier = mock_load_light_curve_from_path
        preloader.current_index_light_curve_pair = IndexLightCurvePair(2, stub_light_curves_dictionary['c.fits'])

        await preloader.load_previous_light_curves()

        assert len(preloader.previous_index_light_curve_pair_deque) == 2
        assert preloader.previous_index_light_curve_pair_deque[0].index == 0
        assert preloader.previous_index_light_curve_pair_deque[0].light_curve == stub_light_curves_dictionary['a.fits']
        assert preloader.previous_index_light_curve_pair_deque[1].index == 1
        assert preloader.previous_index_light_curve_pair_deque[1].light_curve == stub_light_curves_dictionary['b.fits']

    @pytest.mark.asyncio
    async def test_incrementing_preloader_shifts_light_curve_deques(self):
        preloader = Preloader()
        preloader.light_curve_identifiers = ['a.fits', 'b.fits', 'c.fits', 'd.fits']
        stub_light_curves_dictionary = {'a.fits': Mock(), 'b.fits': Mock(), 'c.fits': Mock(), 'd.fits': Mock()}
        preloader.current_index_light_curve_pair = IndexLightCurvePair(1, stub_light_curves_dictionary['b.fits'])
        preloader.previous_index_light_curve_pair_deque = deque(
            [IndexLightCurvePair(0, stub_light_curves_dictionary['a.fits'])])
        preloader.next_index_light_curve_pair_deque = deque(
            [IndexLightCurvePair(2, stub_light_curves_dictionary['c.fits'])])
        preloader.refresh_surrounding_light_curve_loading = AsyncMock()

        new_current = await preloader.increment()

        assert new_current.index == 2
        assert new_current.light_curve == stub_light_curves_dictionary['c.fits']
        assert len(preloader.previous_index_light_curve_pair_deque) == 2
        assert preloader.previous_index_light_curve_pair_deque[-1].index == 1
        assert len(preloader.next_index_light_curve_pair_deque) == 0

    @pytest.mark.asyncio
    async def test_decrementing_preloader_shifts_light_curve_deques(self):
        preloader = Preloader()
        preloader.light_curve_identifiers = ['a.fits', 'b.fits', 'c.fits', 'd.fits']
        stub_light_curves_dictionary = {'a.fits': Mock(), 'b.fits': Mock(), 'c.fits': Mock(), 'd.fits': Mock()}
        preloader.current_index_light_curve_pair = IndexLightCurvePair(1, stub_light_curves_dictionary['b.fits'])
        preloader.previous_index_light_curve_pair_deque = deque(
            [IndexLightCurvePair(0, stub_light_curves_dictionary['a.fits'])])
        preloader.next_index_light_curve_pair_deque = deque(
            [IndexLightCurvePair(2, stub_light_curves_dictionary['c.fits'])])
        preloader.refresh_surrounding_light_curve_loading = AsyncMock()

        new_current = await preloader.decrement()

        assert new_current.index == 0
        assert new_current.light_curve is stub_light_curves_dictionary['a.fits']
        assert len(preloader.next_index_light_curve_pair_deque) == 2
        assert preloader.next_index_light_curve_pair_deque[0].index == 1
        assert len(preloader.previous_index_light_curve_pair_deque) == 0

    @pytest.mark.asyncio
    async def test_incrementing_calls_refresh_of_surrounding_light_curve_loading(self):
        preloader = Preloader()
        preloader.next_index_light_curve_pair_deque = Mock()
        preloader.previous_index_light_curve_pair_deque = Mock()
        preloader.current_index_light_curve_pair = Mock()
        mock_refresh_surrounding_light_curve_loading = AsyncMock()
        preloader.refresh_surrounding_light_curve_loading = mock_refresh_surrounding_light_curve_loading

        _ = await preloader.increment()

        assert mock_refresh_surrounding_light_curve_loading.called

    @pytest.mark.asyncio
    async def test_decrementing_calls_refresh_of_surrounding_light_curve_loading(self):
        preloader = Preloader()
        preloader.next_index_light_curve_pair_deque = Mock()
        preloader.previous_index_light_curve_pair_deque = Mock()
        preloader.current_index_light_curve_pair = Mock()
        mock_refresh_surrounding_light_curve_loading = AsyncMock()
        preloader.refresh_surrounding_light_curve_loading = mock_refresh_surrounding_light_curve_loading

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
        preloader.load_surrounding_light_curves = mock_load_surrounding_light_curves

        with patch.object(module.asyncio, 'create_task') as mock_create_task:
            _ = await preloader.refresh_surrounding_light_curve_loading()

            assert mock_create_task.call_args[0][0] is stub_coroutine

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_factory_from_path_list_loads_light_curves(self):
        stub_light_curves_dictionary = {'a.fits': Mock(), 'b.fits': Mock(), 'c.fits': Mock(), 'd.fits': Mock()}
        def load_light_curve_side_effect(path):
            return stub_light_curves_dictionary[str(path)]
        mock_load_light_curve_from_path = Mock(side_effect=load_light_curve_side_effect)
        Preloader.load_light_curve_from_identifier = mock_load_light_curve_from_path

        paths = [Path(string) for string in ['a.fits', 'b.fits', 'c.fits', 'd.fits']]
        preloader = await Preloader.from_identifier_list(paths, starting_index=1)
        await preloader.running_loading_task

        assert preloader.current_index_light_curve_pair.light_curve == stub_light_curves_dictionary['b.fits']
        assert len(preloader.previous_index_light_curve_pair_deque) == 1
        assert len(preloader.next_index_light_curve_pair_deque) == 2
