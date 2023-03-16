import torch
from torch import nn
from plflow.models.submodules.masked_layer import MaskLinear, MaskConv2d
from plflow.prune.utils import mask_conv_layer_by_magnitude, mask_fc_layer_by_magnitude


class GrowRegMaskConv2d(nn.Conv2d):
    def __init__(self, block_dimension: tuple, sparsity: float, init_args, *args, **kwargs):
        super(GrowRegMaskConv2d, self).__init__(*args, **kwargs)
        self.weight.data = init_args['weight_data']
        if kwargs['bias']:
            self.bias.data = init_args['bias_data']

        self.block_dimension = block_dimension
        self.sparsity = sparsity

        pr_mask = 1 - mask_conv_layer_by_magnitude(self.weight, block_dimension, sparsity, False)
        self.register_buffer('pr_mask', pr_mask)
        self.register_buffer('reg', torch.zeros(1))

    def apply_reg(self):
        self.weight.grad += (self.reg * self.pr_mask * self.weight)

    def update_reg(self, epsilon_lambda):
        self.reg += epsilon_lambda

    def to_finetune_mode(self):
        return MaskConv2d.convert(self, 1 - self.pr_mask)

    @staticmethod
    def convert(m, block_dim, sparsity):
        bias = m.bias is not None
        init_args = {'weight_data': m.weight.data, 'bias_data': m.bias.data if bias else None}
        conv_args = {'in_channels': m.in_channels, 'out_channels': m.out_channels, 'kernel_size': m.kernel_size,
                     'stride': m.stride, 'padding': m.padding, 'groups': m.groups, 'bias': bias}
        new_m = GrowRegMaskConv2d(block_dim, sparsity, init_args, **conv_args)
        return new_m


class GrowRegMaskLinear(nn.Linear):
    def __init__(self, block_dimension: tuple, sparsity: float, init_args, *args, **kwargs):
        super(GrowRegMaskLinear, self).__init__(*args, **kwargs)
        self.weight.data = init_args['weight_data']
        if kwargs['bias']:
            self.bias.data = init_args['bias_data']

        self.block_dimension = block_dimension
        self.sparsity = sparsity

        pr_mask = 1 - mask_fc_layer_by_magnitude(self.weight, block_dimension, sparsity, False)
        self.register_buffer('pr_mask', pr_mask)
        self.register_buffer('reg', torch.zeros(1))

    def apply_reg(self):
        self.weight.grad += (self.reg * self.pr_mask * self.weight)

    def update_reg(self, epsilon_lambda):
        self.reg += epsilon_lambda

    def to_finetune_mode(self):
        return MaskLinear.convert(self, mask=1 - self.pr_mask)

    @staticmethod
    def convert(m, block_dim, sparsity):
        bias = False
        if m.bias is not None:
            bias = True

        init_args = {'weight_data': m.weight.data, 'bias_data': m.bias.data if bias else None}
        fc_args = {'in_features': m.in_features, 'out_features': m.out_features, 'bias': bias}
        new_m = GrowRegMaskLinear(block_dim, sparsity, init_args, **fc_args)
        return new_m
