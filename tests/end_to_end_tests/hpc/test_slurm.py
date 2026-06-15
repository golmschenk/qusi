"""
Tests for the SLURM job module.
"""

from pathlib import Path

from qusi.experimental.hpc.slurm import Job


def test_job_file_creation():
    test_output_path = Path(__file__).parent.joinpath('temporary_slurm_test_job_creation.sh')
    job = Job.new(Path('fake_path.py'), session_name='fake_session',
                  options={'--nodes': 7, '--ntasks-per-node': 1, '--gpus-per-node': 2})
    job.generate_job_script_at_file_path(test_output_path)
    with test_output_path.open('r') as test_output_file_handle:
        test_output_string = test_output_file_handle.read()
    assert '#SBATCH --nodes=7' in test_output_string
    assert '#SBATCH --ntasks-per-node=1' in test_output_string
    assert '--nnodes=7' in test_output_string
    assert '--nproc_per_node=2' in test_output_string
    assert test_output_string.strip().endswith('fake_path.py')
    if test_output_path.exists():
        test_output_path.unlink()
