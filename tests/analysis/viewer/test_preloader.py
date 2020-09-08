from collections import deque
from unittest.mock import Mock

import pytest

from ramjet.analysis.viewer.preloader import Preloader, IndexLightCurvePair


class TestPreloader:
    @pytest.mark.asyncio
    async def test_loading_light_curve_at_index_as_current_calls_loading_with_correct_path(self):
        preloader = Preloader()
        preloader.light_curve_path_list = ['a.fits', 'b.fits', 'c.fits']
        stub_light_curve = Mock()
        mock_load_light_curve_from_path = Mock(return_value=stub_light_curve)
        preloader.load_light_curve_from_path = mock_load_light_curve_from_path
        index = 1

        await preloader.load_light_curve_at_index_as_current(index)

        assert preloader.current_index_light_curve_pair.index == index
        assert preloader.current_index_light_curve_pair.light_curve is stub_light_curve
        assert preloader.load_light_curve_from_path.call_args[0][0] == 'b.fits'

    @pytest.mark.asyncio
    async def test_loading_next_light_curves_loads_from_last_in_next_deque(self):
        preloader = Preloader()
        preloader.light_curve_path_list = ['a.fits', 'b.fits', 'c.fits', 'd.fits']
        stub_light_curves_dictionary = {'a.fits': Mock(), 'b.fits': Mock(), 'c.fits': Mock(), 'd.fits': Mock()}
        def load_light_curve_side_effect(path):
            return stub_light_curves_dictionary[path]
        mock_load_light_curve_from_path = Mock(side_effect=load_light_curve_side_effect)
        preloader.load_light_curve_from_path = mock_load_light_curve_from_path
        preloader.next_index_light_curve_pairs_deque = deque(
            [IndexLightCurvePair(1, stub_light_curves_dictionary['b.fits'])])

        await preloader.load_next_light_curves()

        assert len(preloader.next_index_light_curve_pairs_deque) == 3
        assert preloader.next_index_light_curve_pairs_deque[1].index == 2
        assert preloader.next_index_light_curve_pairs_deque[1].light_curve == stub_light_curves_dictionary['c.fits']
        assert preloader.next_index_light_curve_pairs_deque[2].index == 3
        assert preloader.next_index_light_curve_pairs_deque[2].light_curve == stub_light_curves_dictionary['d.fits']

    @pytest.mark.asyncio
    async def test_loading_next_light_curves_loads_from_current_when_next_deque_is_empty(self):
        preloader = Preloader()
        preloader.light_curve_path_list = ['a.fits', 'b.fits', 'c.fits', 'd.fits']
        stub_light_curves_dictionary = {'a.fits': Mock(), 'b.fits': Mock(), 'c.fits': Mock(), 'd.fits': Mock()}
        def load_light_curve_side_effect(path):
            return stub_light_curves_dictionary[path]
        mock_load_light_curve_from_path = Mock(side_effect=load_light_curve_side_effect)
        preloader.load_light_curve_from_path = mock_load_light_curve_from_path
        preloader.current_index_light_curve_pair = IndexLightCurvePair(1, stub_light_curves_dictionary['b.fits'])

        await preloader.load_next_light_curves()

        assert len(preloader.next_index_light_curve_pairs_deque) == 2
        assert preloader.next_index_light_curve_pairs_deque[0].index == 2
        assert preloader.next_index_light_curve_pairs_deque[0].light_curve == stub_light_curves_dictionary['c.fits']
        assert preloader.next_index_light_curve_pairs_deque[1].index == 3
        assert preloader.next_index_light_curve_pairs_deque[1].light_curve == stub_light_curves_dictionary['d.fits']

    @pytest.mark.asyncio
    async def test_loading_previous_light_curves_loads_from_first_in_previous_deque(self):
        preloader = Preloader()
        preloader.light_curve_path_list = ['a.fits', 'b.fits', 'c.fits', 'd.fits']
        stub_light_curves_dictionary = {'a.fits': Mock(), 'b.fits': Mock(), 'c.fits': Mock(), 'd.fits': Mock()}
        def load_light_curve_side_effect(path):
            return stub_light_curves_dictionary[path]
        mock_load_light_curve_from_path = Mock(side_effect=load_light_curve_side_effect)
        preloader.load_light_curve_from_path = mock_load_light_curve_from_path
        preloader.previous_index_light_curve_pairs_deque = deque(
            [IndexLightCurvePair(2, stub_light_curves_dictionary['c.fits'])])

        await preloader.load_previous_light_curves()

        assert len(preloader.previous_index_light_curve_pairs_deque) == 3
        assert preloader.previous_index_light_curve_pairs_deque[0].index == 0
        assert preloader.previous_index_light_curve_pairs_deque[0].light_curve == stub_light_curves_dictionary['a.fits']
        assert preloader.previous_index_light_curve_pairs_deque[1].index == 1
        assert preloader.previous_index_light_curve_pairs_deque[1].light_curve == stub_light_curves_dictionary['b.fits']


    @pytest.mark.asyncio
    async def test_loading_previous_light_curves_loads_from_first_in_previous_deque(self):
        preloader = Preloader()
        preloader.light_curve_path_list = ['a.fits', 'b.fits', 'c.fits', 'd.fits']
        stub_light_curves_dictionary = {'a.fits': Mock(), 'b.fits': Mock(), 'c.fits': Mock(), 'd.fits': Mock()}
        def load_light_curve_side_effect(path):
            return stub_light_curves_dictionary[path]
        mock_load_light_curve_from_path = Mock(side_effect=load_light_curve_side_effect)
        preloader.load_light_curve_from_path = mock_load_light_curve_from_path
        preloader.previous_index_light_curve_pairs_deque = deque(
            [IndexLightCurvePair(2, stub_light_curves_dictionary['c.fits'])])

        await preloader.load_previous_light_curves()

        assert len(preloader.previous_index_light_curve_pairs_deque) == 3
        assert preloader.previous_index_light_curve_pairs_deque[0].index == 0
        assert preloader.previous_index_light_curve_pairs_deque[0].light_curve == stub_light_curves_dictionary['a.fits']
        assert preloader.previous_index_light_curve_pairs_deque[1].index == 1
        assert preloader.previous_index_light_curve_pairs_deque[1].light_curve == stub_light_curves_dictionary['b.fits']

    @pytest.mark.asyncio
    async def test_loading_previous_light_curves_loads_from_current_when_previous_deque_is_empty(self):
        preloader = Preloader()
        preloader.light_curve_path_list = ['a.fits', 'b.fits', 'c.fits', 'd.fits']
        stub_light_curves_dictionary = {'a.fits': Mock(), 'b.fits': Mock(), 'c.fits': Mock(), 'd.fits': Mock()}
        def load_light_curve_side_effect(path):
            return stub_light_curves_dictionary[path]
        mock_load_light_curve_from_path = Mock(side_effect=load_light_curve_side_effect)
        preloader.load_light_curve_from_path = mock_load_light_curve_from_path
        preloader.current_index_light_curve_pair = IndexLightCurvePair(2, stub_light_curves_dictionary['c.fits'])

        await preloader.load_previous_light_curves()

        assert len(preloader.previous_index_light_curve_pairs_deque) == 2
        assert preloader.previous_index_light_curve_pairs_deque[0].index == 0
        assert preloader.previous_index_light_curve_pairs_deque[0].light_curve == stub_light_curves_dictionary['a.fits']
        assert preloader.previous_index_light_curve_pairs_deque[1].index == 1
        assert preloader.previous_index_light_curve_pairs_deque[1].light_curve == stub_light_curves_dictionary['b.fits']
