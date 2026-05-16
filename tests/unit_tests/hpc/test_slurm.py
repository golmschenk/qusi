"""
Tests for the SLURM job module.
"""
from qusi.experimental.hpc.slurm import Job


def test_adding_option():
    job = Job()
    job.add_option('--fake-option-key', 10)
    assert job.options['--fake-option-key'] == '10'

def test_adding_option_for_number_of_nodes_under_short_name():
    job = Job()
    job.add_option('-N', 7)
    assert job.options['--nnodes'] == '7'

def test_adding_multiple_options():
    job = Job()
    job.add_options({'--fake-option-key0': 'a', '--fake-option-key1': 7})
    assert job.options['--fake-option-key0'] == 'a'
    assert job.options['--fake-option-key1'] == '7'
