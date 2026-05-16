"""
A module for running SLURM jobs.
"""
from pathlib import Path

from typing import Self, TextIO


class MissingRequiredJobOptionException(Exception):
    """
    Exception for missing job options that are required by the SLURM job configuration.
    """
    def __init__(self, option_key):
        self.message = f'Job option `{option_key}` is required.'
        super().__init__(self.message)


class Job:
    """
    A class to represent launch SLURM jobs.
    """
    def __init__(self):
        self.options: dict[str, str] = {}

    @classmethod
    def new(cls, options: dict[str, str] | None = None) -> Self:
        """
        The default constructor for new jobs.

        :param options: The options to pass to the SLURM job.
        :return: The SLURM job.
        """
        if options is None:
            options = {}
        instance = cls()
        instance.add_options(options)
        return instance

    def add_options(self, options: dict[str, str | int]) -> None:
        """
        Add multiple options as a dictionary.

        :param options: The dictionary of options to add.
        """
        for option_key, option_value in options.items():
            self.add_option(option_key, option_value)

    def add_option(self, option_key: str, option_value: str | int) -> None:
        """
        Adds a SLURM batch option to the produced job.

        :param option_key: The key of the option to add.
        :param option_value: The value of the option to add.
        """
        if isinstance(option_value, int):
            option_value = str(option_value)
        if option_key == '-N':
            option_key = '--nnodes'
        self.options[option_key] = option_value

    def generate_job_script_at_file_path(self, path: Path) -> None:
        """
        Generates the SLURM batch job script at the given path.

        :param path: The path to generate the job script at.
        """
        with path.open(mode='w') as file_handle:
            self.write_job_script_in_file_handle(file_handle=file_handle)

    def write_job_script_in_file_handle(self, file_handle: TextIO) -> None:
        """
        Generate the SLURM batch job script at the file handle.

        :param file_handle: The file handle to write the job script to.
        """
        self.write_hashbang_to_file_handle(file_handle)
        self.write_job_options_to_file_handle(file_handle)

    def write_hashbang_to_file_handle(self, file_handle: TextIO) -> None:
        """
        Writes the hashbang to the file handle.

        :param file_handle: The file handle to write to.
        """
        file_handle.write(f'#!/bin/bash\n')
        file_handle.write(f'\n')

    def write_job_options_to_file_handle(self, file_handle: TextIO) -> None:
        """
        Writes the SLURM batch options to the file handle.

        :param file_handle: The file handle to write to.
        """
        for option_name, option_value in self.options:
            file_handle.write(fr'#SBATCH {option_name}={option_value}\n')

    def write_torch_distributed_call_to_file_handle(self, file_handle: TextIO) -> None:
        """
        Writes the PyTorch distributed call to the file handle.

        :param file_handle: The file handle to write to.
        """
        if self.options.get('--nnodes') is None:
            raise MissingRequiredJobOptionException('--nnodes')
        if self.options.get('--nproc_per_node') is None:
            raise MissingRequiredJobOptionException('--nproc_per_node')
        number_of_nodes = int(self.options['--nnodes'])
        training_processes_per_node = int(self.options['--ntasks-per-node'])
        script_path = Path('scripts/1m_sqlite_train_session.py')
        file_handle.write(
            f'python -m torch.distributed.run \\\n'
            f'--nnodes {number_of_nodes} \\\n'
            f'--nproc_per_node {training_processes_per_node} \\\n'
            f'--rdzv_id $RANDOM \\\n'
            f'--rdzv_backend c10d \\\n'
            f'--rdzv_endpoint $head_node_hostname \\\n'
            f'{script_path}'
        )
