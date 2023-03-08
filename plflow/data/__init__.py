import pl_bolts.datamodules as dms
import torchvision

from pl_bolts.transforms.dataset_normalizations import cifar10_normalization, imagenet_normalization

from plflow.data.datamodules.cifar import CIFARDataModule
from plflow.data.datamodules.glue import GLUEDataModule
from plflow.data.datamodules.tarimgnet import TarImgNetDataModule
from plflow.data.transforms.cifar100 import cifar100_normalization
from plflow.data.transforms.rgb2tensor import RGBToTensor


def cifar10_datamodule(num_worker, batch_size, data_dir, drop_last):
    return CIFARDataModule(
        num_worker, batch_size, data_dir, drop_last,
        cls=torchvision.datasets.CIFAR10, normalization=cifar10_normalization
    )


def cifar100_datamodule(num_worker, batch_size, data_dir, drop_last):
    return CIFARDataModule(
        num_worker, batch_size, data_dir, drop_last,
        cls=torchvision.datasets.CIFAR100, normalization=cifar100_normalization
    )


def glue_datamodule(
        model_name_or_path: str,
        task_name: str = "mrpc",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
):
    return GLUEDataModule(
        model_name_or_path,
        task_name,
        max_seq_length,
        train_batch_size,
        eval_batch_size,
        num_workers,
    )


def imgnet_datamodule(*args, **kwargs):
    dm = dms.ImagenetDataModule(*args, **kwargs)

    dm.prepare_data()
    dm.setup()

    dm._train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            RGBToTensor(),
            imagenet_normalization()
        ]
    )

    dm._val_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224 + 32),
            torchvision.transforms.CenterCrop(224),
            RGBToTensor(),
            imagenet_normalization()
        ]
    )

    dm._test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224 + 32),
            torchvision.transforms.CenterCrop(224),
            RGBToTensor(),
            imagenet_normalization()
        ]
    )

    return dm


def tar_imgnet_datamodule(*args, **kwargs):
    dm = TarImgNetDataModule(*args, **kwargs)
    dm.prepare_data()
    dm.setup()

    return dm
