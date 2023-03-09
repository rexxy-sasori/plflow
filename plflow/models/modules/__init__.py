from torch import nn


def convert_module(m: nn.Module, conv_converter=None, linear_converter=None, *args, **kwargs):
    if isinstance(m, nn.Conv2d):
        return conv_converter(m, *args, **kwargs)
    elif isinstance(m, nn.Linear):
        return linear_converter(m, *args, **kwargs)
    else:
        raise ValueError('You can not mask a module that is neither linear nor Conv')
