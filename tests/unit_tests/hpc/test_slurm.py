"""
Tests for the SLURM job module.
"""
import pytest
from io import StringIO
from pathlib import Path

from qusi.experimental.hpc.slurm import Job, MissingRequiredJobOptionException


def test_adding_option():
    job = Job(Path('sessions/fake_session'), Path('fake_path.py'))
    job.add_option('--fake-option-key', 10)
    assert job.options['--fake-option-key'] == '10'

def test_adding_option_for_number_of_nodes_under_short_name():
    job = Job(Path('sessions/fake_session'), Path('fake_path.py'))
    job.add_option('-N', 7)
    assert job.options['--nodes'] == '7'

def test_adding_multiple_options():
    job = Job(Path('sessions/fake_session'), Path('fake_path.py'))
    job.add_options({'--fake-option-key0': 'a', '--fake-option-key1': 7})
    assert job.options['--fake-option-key0'] == 'a'
    assert job.options['--fake-option-key1'] == '7'

def test_torch_distributed_call_generation_requires_number_of_nodes_option():
    task_path = Path('fake_path.py')
    job = Job(Path('sessions/fake_session'), task_path)
    string_io = StringIO()
    with pytest.raises(MissingRequiredJobOptionException, match='--nodes'):
        job.write_torch_distributed_call_to_file_handle(string_io)

def test_torch_distributed_call_generation_requires_tasks_per_node_option():
    task_path = Path('fake_path.py')
    job = Job(Path('sessions/fake_session'), task_path)
    job.add_option('--nodes', 7)
    string_io = StringIO()
    with pytest.raises(MissingRequiredJobOptionException, match='--ntasks-per-node'):
        job.write_torch_distributed_call_to_file_handle(string_io)

def test_torch_distributed_call_uses_script_path():
    task_path = Path('fake_path.py')
    job = Job(Path('sessions/fake_session'), task_path)
    job.add_option('--nodes', 7)
    job.add_option('--ntasks-per-node', 2)
    string_io = StringIO()
    job.write_torch_distributed_call_to_file_handle(string_io)
    assert string_io.getvalue().strip().endswith(str(task_path))
