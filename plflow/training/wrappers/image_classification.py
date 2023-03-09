from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics

from plflow.config_parsing.parsers import parse_optimization_config
from plflow.config_parsing import usr_config
from torch import nn


class ImageClassificationWrapper(pl.LightningModule):
    """
    This class wraps a nn.Module for image classification (e.g. ResNet) that is
    later to be used for training

    Where to use it:
    ...
    usr_config = get_usr_config_from_cmd_line(args) # from /rex/de/my/config.yaml
    datamodule = parse_data_module(usr_config) # from parsers
    model = parse_model(usr_config)
    plmodule = ImgClassificationWrapper(model, usr_config, save_usr_config=True)

    trainer = pl.Trainer(logger, callbacks, strategy)
    # check pytorch lightning for what logger, callbacks, and strategy are

    trainer.fit(plmodule, datamodule=datamodule) # train the model
    trainer.test(plmodule, datamodule=datamodule) # test the model

    final output:

    Test metric:           DataLoader 0
    ----------------------------------------
    accuracy:             0.824248294323424
    sparsity:             0.341323423483242
    ...
    """

    def __init__(
            self,
            model: nn.Module,
            usr_config: usr_config.UsrConfigs,
            save_usr_config=False
    ):
        super(ImageClassificationWrapper, self).__init__()
        self.module = model
        self.usr_config = usr_config

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

        if save_usr_config:
            self.save_hyperparameters('usr_config')

    def configure_optimizers(self):
        try:
            optimizer_config = self.usr_config.optimizer
            lr_scheduler_config = self.usr_config.lr_scheduler

            optimizer, lr_scheduler = parse_optimization_config(
                self.module, optimizer_config, lr_scheduler_config,
                optimlib=torch.optim, lr_schedulerlib=torch.optim.lr_scheduler
            )

            if hasattr(optimizer_config, 'label_smoothing'):
                self.label_smoothing = optimizer_config.label_smoothing
            else:
                self.label_smoothing = 0

        except AttributeError:
            # fall back to inference mode, just some random optimizers
            optimizer = torch.optim.SGD(self.parameters(), lr=0)
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1], gamma=0.1)
            self.label_smoothing = 0

        return [optimizer], [lr_scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def forward(self, *args, **kwargs) -> Any:
        return self.module(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        train_loss = F.cross_entropy(y_hat, y, label_smoothing=self.label_smoothing)

        self.train_acc(y_hat, y)

        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log('train_loss', train_loss, on_step=True, on_epoch=True)

        return {'loss': train_loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)

        self.val_acc(y_hat, y)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True, logger=True)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)

        self.test_acc(y_hat, y)
        self.log('test_acc', self.test_acc, on_step=True, on_epoch=True, logger=True)
        self.log('test_loss', test_loss, on_step=True, on_epoch=True, logger=True)
