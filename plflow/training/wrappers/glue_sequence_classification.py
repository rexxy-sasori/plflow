from typing import Optional

import datasets
import pytorch_lightning as pl
import torch
import transformers
from torch import nn

from plflow.analysis import get_sparsity
from plflow.config_parsing import usr_config
from plflow.config_parsing.parsers import parse_optimization_config
from plflow.training.utils import run_glue_benchmark


class GlueSequenceClassificationWrapper(pl.LightningModule):
    """
    This class wraps a nn.Module for glue benchmark (e.g. BERT) that is
    later to be used for training

    Where to use it:
    ...
    usr_config = get_usr_config_from_cmd_line(args)
    datamodule = parse_data_module(usr_config) # from parsers
    model = parse_model(usr_config)
    plmodule = GlueSequenceClassificationWrapper(model, usr_config, ..., save_usr_config=True)

    trainer = pl.Trainer(logger, callbacks, strategy)
    # check pytorch lightning for what logger, callbacks, and strategy are
    # There is an article "fintuning bert for the glue benchmark"
    # Checking it out would be helpful

    trainer.fit(plmodule, datamodule=datamodule) # train the model
    trainer.test(plmodule, datamodule=datamodule) # test the model
    ...
    """

    def __init__(
            self,
            model: nn.Module,
            usr_config: usr_config.UsrConfigs,
            task_name: str,
            num_labels: int,
            eval_splits: Optional[int],
            save_usr_config: bool = False
    ):
        super(GlueSequenceClassificationWrapper, self).__init__()
        self.module = model
        self.usr_config = usr_config
        self.metric = datasets.load_metric("glue", task_name)
        self.num_labels = num_labels
        self.eval_splits = eval_splits

        if save_usr_config:
            self.save_hyperparameters('usr_config')

    def forward(self, **inputs):
        return self.module(**inputs)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def training_step(self, batch, batch_idx):
        outputs = self.module(**batch)
        loss = outputs[0]
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.module(**batch)
        val_loss, logits = outputs[:2]

        if self.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, logger=True)
        return {"loss": val_loss, "preds": preds, "labels": labels}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.module(**batch)
        val_loss, logits = outputs[:2]

        if self.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]
        self.log('test_loss', val_loss, on_step=True, on_epoch=True, logger=True)

        sparsity = get_sparsity(self.module)['sparsity']
        self.log('sparsity', sparsity, on_step=False, on_epoch=True, logger=True)

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        run_glue_benchmark(self, outputs, "{}_{}", "val_loss_{}", "val_loss")

    def configure_optimizers(self):
        optimizer, lr_scheduler = parse_optimization_config(
            model=self.module, optimizer_config=self.optimizer_config,
            lr_scheduler_config=self.lr_scheduler_config,
            optimlib=transformers, lr_schedulerlib=transformers
        )

        return [optimizer], [lr_scheduler]
