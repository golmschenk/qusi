"""
A module for running SLURM jobs.
"""
import datetime
import shutil
import subprocess
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

    def __init__(self, session_directory: Path, torch_task_script_path: Path):
        self.session_directory: Path = session_directory
        self.original_torch_task_script_path: Path = torch_task_script_path
        self.session_torch_task_script_path: Path = session_directory.joinpath(torch_task_script_path.name)
        self.session_shell_task_script_path: Path = session_directory.joinpath('job_script.sh')
        self.options: dict[str, str] = {}

    @classmethod
    def new(
            cls,
            torch_task_script_path: Path,
            session_name: str,
            sessions_root_directory: Path | None = None,
            options: dict[str, str | int] | None = None
    ) -> Self:
        """
        The default constructor for new jobs.

        :param torch_task_script_path: The path to the torch job script to run from the SLURM job.
        :param session_name: The name of the session.
        :param sessions_root_directory: The root sessions directory that contains all sessions (defaults to `sessions`).
        :param options: The options to pass to the SLURM job.
        :return: The SLURM job.
        """
        if sessions_root_directory is None:
            sessions_root_directory = Path('sessions')
        if options is None:
            options = {}
        session_directory = cls.get_session_directory(session_name, sessions_root_directory)
        instance = cls(session_directory=session_directory, torch_task_script_path=torch_task_script_path)
        instance.add_options(options)
        return instance

    @classmethod
    def get_session_directory(cls, session_name: str, sessions_root_directory: Path) -> Path:
        """
        Get a session directory with a datetime prefix.

        :param session_name: The name of the session.
        :param sessions_root_directory: The root directory that contains all the sessions.
        :return: The path to the session directory.
        """
        datetime_string = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        session_directory = sessions_root_directory.joinpath(f'{datetime_string}_{session_name}')
        return session_directory

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
            option_key = '--nodes'
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
        self.write_torch_distributed_call_to_file_handle(file_handle)

    @staticmethod
    def write_hashbang_to_file_handle(file_handle: TextIO) -> None:
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
        for option_name, option_value in self.options.items():
            file_handle.write(f'#SBATCH {option_name}={option_value}\n')

    def write_torch_distributed_call_to_file_handle(self, file_handle: TextIO) -> None:
        """
        Writes the PyTorch distributed call to the file handle.

        :param file_handle: The file handle to write to.
        """
        for required_job_option in ['--nodes', '--ntasks-per-node', '--gpus-per-node']:
            if self.options.get(required_job_option) is None:
                raise MissingRequiredJobOptionException(required_job_option)
        number_of_nodes = int(self.options['--nodes'])
        training_processes_per_node = int(self.options['--gpus-per-node']) // int(self.options['--ntasks-per-node'])
        file_handle.write(
            f'srun python -m torch.distributed.run \\\n'
            f'--nnodes={number_of_nodes} \\\n'
            f'--nproc_per_node={training_processes_per_node} \\\n'
            f'--rdzv_id=$RANDOM \\\n'
            f'--rdzv_backend=c10d \\\n'
            f'--rdzv_endpoint=$head_node_hostname \\\n'
            f'{self.session_torch_task_script_path}\n'
        )

    def prepare_session_directory(self) -> None:
        """
        Prepare the session directory for running, including creation of the directory and moving the scripts in.
        """
        self.session_directory.mkdir(parents=True)
        shutil.copyfile(self.original_torch_task_script_path, self.session_torch_task_script_path)
        self.generate_job_script_at_file_path(self.session_shell_task_script_path)

    def run(self) -> None:
        """
        Executes the job.
        """
        self.prepare_session_directory()
        subprocess.run(['sbatch', f'{self.session_shell_task_script_path}'])
