import os


def ensure_torchrun_environment_variables():
    if 'RANK' not in os.environ:
        # The script was not called with `torchrun` and environment variables need to be set manually.
        os.environ['RANK'] = str(0)
        os.environ['LOCAL_RANK'] = str(0)
        os.environ['WORLD_SIZE'] = str(1)
        os.environ['LOCAL_WORLD_SIZE'] = str(1)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '35728'
        os.environ['TORCHELASTIC_RESTART_COUNT'] = str(0)
