import torch
from nnutils.cnn_complexity_analyzer.profile import profile_compute_layers
from torch import nn

from plflow.utils import is_linear_transform_layer


def print_gemm_param(model, inputs, stdout_print=True):
    results, _ = profile_compute_layers(model, inputs)
    ret = {}
    for name, result in results.items():
        if result.is_conv:
            cout, cin, hk, wk = result.weight.shape
            M, K = cout, cin * hk * wk
            N = result.output_dim[-1] * result.output_dim[-2]
            N = N.item()
        else:
            M, K = result.weight.shape
            N = 1

        if stdout_print:
            print('{' + f"{M}, {K}, {N}" + '},')
        ret[name] = (M, K, N)
    return ret


def get_num_params(model):
    if isinstance(model, nn.Module):
        ret = sum(
            [p.numel() for n, p in model.state_dict().items() if (p.dim() == 4 or p.dim() == 2) and 'weight' in n]
        )
    else:
        ret = sum([p.weight.data.numel() for p in model.values()])
    return ret


def get_sparsity(inputs):
    if isinstance(inputs, nn.Module):
        num_ones, num_total = _get_sparsity_nn_module(inputs)
    else:
        num_ones, num_total = _get_sparsity_dict_layers(inputs)

    density = num_ones / num_total
    sparsity = 1 - density
    return {'num_ones': num_ones, 'num_total': num_total, 'sparsity': sparsity, 'density': density}


def _get_sparsity_dict_layers(targets):
    num_ones = 0
    num_total = 0
    for name, target in targets.items():
        if hasattr(target, 'mask'):
            one_map = (target.mask.data != 0).float()
        else:
            one_map = (target.weight.data != 0).float()
        num_ones += one_map.sum().item()
        num_total += one_map.numel()

    return num_ones, num_total


def _get_sparsity_nn_module(model):
    num_ones = 0
    num_total = 0
    for n, m in model.named_modules():
        if is_linear_transform_layer(m):
            if hasattr(m, 'mask'):
                num_ones += m.mask.sum().item()
            else:
                one_mask = (m.weight.data != 0).float()
                num_ones += one_mask.sum().item()

            num_total += m.weight.data.numel()
    return num_ones, num_total


def get_pr_over_kp(module: nn.Module, mask: torch.Tensor):
    """
    Given a mask dividing the model into two sets, this method computes the ratio of the pruned weights' power to the
    remaining weights
    """
    if get_sparsity(module)['sparsity'] == 0:
        return 0
    else:
        kp_mask = mask
        pr_mask = 1 - kp_mask
        pr_weight_norm = torch.norm(pr_mask * module.weight.data) ** 2
        kp_weight_norm = torch.norm(kp_mask * module.weight.data) ** 2
        ret = torch.sqrt(pr_weight_norm / kp_weight_norm)
        return ret
