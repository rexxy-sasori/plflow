from typing import List

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import strategies as pl_strategies

from plflow.utils import debug_msg


def execute(
        plmodule: pl.LightningModule,
        datamodule: pl.LightningDataModule,

        num_epochs: int,
        ckpt_path: str,

        logger: pl_loggers.logger.Logger,
        strategy: [str, pl_strategies.Strategy],
        callbacks: List[pl.callbacks.Callback],

        verbose: bool,
):
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        accumulate_grad_batches=1,
        benchmark=True,
        devices=-1,
        logger=logger,
        strategy=strategy,
        callbacks=callbacks,
    )

    try:
        debug_msg(f"loading ckpt from: {ckpt_path}", verbose)
        ckpt = torch.load(ckpt_path)
        try:
            plmodule.load_state_dict(ckpt['state_dict'])
        except Exception:
            msg = """ Error encountered, probably mismatched keys while 
            loading for pl-wrapped model, retrying with module removed
            """
            debug_msg(msg, verbose)
            plmodule.module.load_state_dict(ckpt['state_dict'])

        debug_msg(f"finished loading model states from {ckpt_path} !! Starting immediate testing", verbose)
        trainer.test(model=plmodule, datamodule=datamodule)
    except FileNotFoundError:
        debug_msg(f'ckpt not found at {ckpt_path}, start train the model', verbose)
        trainer.fit(model=plmodule, datamodule=datamodule)
    finally:
        trainer.test(model=plmodule, datamodule=datamodule)
