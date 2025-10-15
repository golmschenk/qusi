from typing import Any
from typing_extensions import override

import lightning
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import convert_inf


class ProgressBar(TQDMProgressBar):
    @override
    def on_train_epoch_start(self, trainer: lightning.pytorch.Trainer, *_: Any) -> None:
        if self._leave:
            self.train_progress_bar = self.init_train_tqdm()
        total = convert_inf(self.total_train_batches)
        self.train_progress_bar.reset()
        self.train_progress_bar.total = total
        self.train_progress_bar.initial = 0
        self.train_progress_bar.set_description(f"Cycle {trainer.current_epoch}")
