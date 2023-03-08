from typing import Optional

import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader

from plflow.data.transforms.rgb2tensor import RGBToTensor


class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, num_worker, batch_size, data_dir, drop_last, cls, normalization):
        super(CIFARDataModule, self).__init__()
        self.num_worker = num_worker
        self.batch_size = batch_size
        self.root_dir = data_dir
        self.drop_last = drop_last
        self.cls = cls

        self.aug_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                RGBToTensor(),
                normalization(),
            ]
        )

        self.base_transform = torchvision.transforms.Compose(
            [
                RGBToTensor(),
                normalization(),
            ]
        )

    def prepare_data(self) -> None:
        self.cls(self.root_dir, train=True, download=True, transform=self.aug_transform)
        self.cls(self.root_dir, train=False, download=True, transform=self.base_transform)

    def setup(self, stage: Optional[str] = None) -> None:
        self.cifar_train = self.cls(self.root_dir, train=True, download=True, transform=self.aug_transform)
        self.cifar_test = self.cls(self.root_dir, train=False, download=True, transform=self.base_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_worker)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar_test, batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_worker, drop_last=self.drop_last)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar_test, batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_worker, drop_last=self.drop_last)
