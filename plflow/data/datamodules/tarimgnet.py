import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader

from plflow.data.transforms.rgb2tensor import RGBToTensor

from typing import Callable
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from plflow.data.datasets.tarimgnetfolder import TarImageFolder


class TarImgNetDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_tar_path,
                 val_tar_path,
                 num_imgs_per_val_class: int = 50,
                 image_size: int = 224,
                 num_workers: int = 0,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 pin_memory: bool = True,
                 drop_last: bool = False,
                 *args, **kwargs):
        super(TarImgNetDataModule, self).__init__(*args, **kwargs)
        self.train_tar_path = train_tar_path
        self.val_tar_path = val_tar_path
        self.num_imgs_per_val_class = num_imgs_per_val_class
        self.image_size = image_size
        self.dims = (3, self.image_size, self.image_size)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.num_samples = 1281167 - self.num_imgs_per_val_class * self.num_classes

    @property
    def num_classes(self) -> int:
        return 1000

    def train_dataloader(self):
        transforms = self.train_transform() if self.train_transforms is None else self.train_transforms
        dataset = TarImageFolder(
            self.train_tar_path,
            split='train',
            transforms=transforms,
            num_val_images_per_class=self.num_imgs_per_val_class,
            num_classes=self.num_classes,
        )

        loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

        return loader

    def val_dataloader(self) -> DataLoader:
        transforms = self.val_transform() if self.val_transforms is None else self.val_transforms

        dataset = TarImageFolder(
            self.train_tar_path,
            split='val',
            transforms=transforms,
            num_val_images_per_class=self.num_imgs_per_val_class,
            num_classes=self.num_classes,
        )

        loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def test_dataloader(self):
        transforms = self.val_transform() if self.test_transforms is None else self.test_transforms

        dataset = TarImageFolder(
            self.val_tar_path,
            split='test',
            transforms=transforms,
            num_val_images_per_class=-1,
            num_classes=self.num_classes,
        )

        loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

        return loader

    def train_transform(self) -> Callable:
        preprocessing = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(self.image_size),
                torchvision.transforms.RandomHorizontalFlip(),
                RGBToTensor(),
                imagenet_normalization(),
            ]
        )

        return preprocessing

    def val_transform(self) -> Callable:
        preprocessing = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(self.image_size + 32),
                torchvision.transforms.CenterCrop(self.image_size),
                RGBToTensor(),
                imagenet_normalization(),
            ]
        )
        return preprocessing
