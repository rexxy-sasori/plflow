from typing import Callable

from torch import nn


def convert_module(m: nn.Module, converter_func: Callable, *args, **kwargs):
    assert isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)
    assert converter_func is not None

    return converter_func(m, *args, **kwargs)
