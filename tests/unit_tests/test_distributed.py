import os
from unittest.mock import Mock, patch

from qusi.internal import distributed


def test_distributed_helpers_without_data_loader_workers():
    with (patch.dict(os.environ, {'RANK': '1', 'WORLD_SIZE': '4'}),
          patch('torch.utils.data.get_worker_info', return_value=None)):
        assert distributed.get_world_rank() == 1
        assert distributed.get_world_size() == 4
        assert distributed.get_local_worker_id() == 0
        assert distributed.get_local_worker_count() == 1
        assert distributed.get_world_worker_id() == 1
        assert distributed.get_world_worker_count() == 4


def test_distributed_helpers_with_data_loader_workers():
    worker_info = Mock(id=2, num_workers=3)
    with (patch.dict(os.environ, {'RANK': '1', 'WORLD_SIZE': '4'}),
          patch('torch.utils.data.get_worker_info', return_value=worker_info)):
        assert distributed.get_local_worker_id() == 2
        assert distributed.get_local_worker_count() == 3
        assert distributed.get_world_worker_id() == 5
        assert distributed.get_world_worker_count() == 12
