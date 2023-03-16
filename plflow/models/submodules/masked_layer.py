import torch
from torch import nn
import torch.nn.functional as F


class MaskConv2d(nn.Conv2d):
    def __init__(self, mask, init_args, *args, **kwargs):
        super(MaskConv2d, self).__init__(*args, **kwargs)
        self.weight.data = init_args['weight_data']
        if kwargs['bias']:
            self.bias.data = init_args['bias_data']

        self.register_buffer('mask', mask)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        masked_weight = self.weight.data.mul(self.mask)
        out = F.conv2d(input, masked_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out

    @staticmethod
    def convert(m, mask):
        bias = m.bias is not None
        init_args = {'weight_data': m.weight.data, 'bias_data': m.bias.data if bias else None}
        conv_args = {'in_channels': m.in_channels, 'out_channels': m.out_channels, 'kernel_size': m.kernel_size,
                     'stride': m.stride, 'padding': m.padding, 'groups': m.groups, 'bias': bias}
        new_m = MaskConv2d(mask, init_args, **conv_args)
        return new_m


class MaskLinear(nn.Linear):
    def __init__(self, mask, init_args, *args, **kwargs):
        super(MaskLinear, self).__init__(*args, **kwargs)
        self.weight.data = init_args['weight_data']
        if kwargs['bias']:
            self.bias.data = init_args['bias_data']

        self.register_buffer('mask', mask)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        masked_weight = self.weight.data.mul(self.mask)
        out = F.linear(input, masked_weight, self.bias)
        return out

    @staticmethod
    def convert(m, mask):
        bias = False
        if m.bias is not None:
            bias = True

        init_args = {'weight_data': m.weight.data, 'bias_data': m.bias.data if bias else None}
        fc_args = {'in_features': m.in_features, 'out_features': m.out_features, 'bias': bias}
        new_m = MaskLinear(mask, init_args, **fc_args)
        return new_m