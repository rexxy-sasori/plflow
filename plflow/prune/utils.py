import itertools
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from plflow.analysis import get_sparsity
from plflow.utils import (
    matrix_to_blocks,
    blocks_to_matrix,
    is_linear_transform_layer,
    strip_module_in_module_name,
)


def get_block_dim(block_policy, name, module):
    ret = block_policy.get(strip_module_in_module_name(name))
    if ret is not None:
        return ret
    else:
        ret = block_policy.get(tuple(module.weight.data.shape))
        if ret is not None:
            return ret
        else:
            raise KeyError('block policy is not hashed by neither module name nor module shape')


def exp_pruning_schedule(current_epoch, s_f, s_i=0, pruning_interval=2, sparsity_step=2) -> float:
    exp_term = (s_f / sparsity_step) * ((current_epoch + pruning_interval) // pruning_interval)
    s_t = s_f + (s_i - s_f) * np.exp(-exp_term)
    return s_t


def mask_fc_layer_by_magnitude(weight, block_dims, alpha, load_balance):
    br, bc = block_dims
    alpha = alpha

    blocks, num_blocks_row, num_blocks_col = matrix_to_blocks(weight, br, bc)
    weight_blocks_reshape = blocks.reshape(num_blocks_row, num_blocks_col, br, bc)

    if not load_balance:
        score = torch.norm(blocks, dim=(-2, -1)) ** 2
        _, sorted_indices = torch.sort(score)

        mask = torch.ones_like(blocks)
        mask[sorted_indices[0:int(num_blocks_row * num_blocks_col * alpha)], :, :] = 0
        mask = blocks_to_matrix(mask, num_blocks_row, num_blocks_col, br, bc)
    else:
        mask = torch.ones_like(weight_blocks_reshape)

    return mask


def mask_conv_layer_by_magnitude(weight, block_dims, alpha, load_balance):
    cout, cin, hk, wk = weight.shape
    unroll_weight = weight.reshape(cout, cin * hk * wk)
    unroll_mask = mask_fc_layer_by_magnitude(unroll_weight, block_dims, alpha, load_balance)
    return unroll_mask.reshape(cout, cin, hk, wk)


def extract_target_cls_layers(model: nn.Module, cls):
    return {name: module for name, module in model.named_modules() if isinstance(module, cls)}


def find_targets(model: nn.Module, prune_first_layer, prune_last_layer, prune_fc_layer=False):
    linear_transform_modules = {
        name: module for name, module in model.named_modules()
        if is_linear_transform_layer(module)
    }

    target_layers = {}
    for idx, (name, module) in enumerate(linear_transform_modules.items()):
        if not prune_fc_layer:
            continue
        if idx == 0 and not prune_first_layer:
            continue
        if idx == len(linear_transform_modules) - 1 and not prune_last_layer:
            continue

        target_layers[name] = module

    return target_layers


def local_prune_model(targets: Dict[str, nn.Module], pruning_rate, block_policy, score_func):
    for name, module in targets.items():
        weight = module.weight.data
        blockdim = get_block_dim(block_policy, name, module)
        if isinstance(module, nn.Conv2d):
            cout, cin, hk, wk = weight.shape
            weight = weight.reshape(cout, cin * hk * wk)

        weight_blocks, num_blocks_row, num_blocks_col = matrix_to_blocks(weight, *blockdim)
        block_score, block_score_sep = score_func(weight_blocks)
        num_blocks_rm = int(num_blocks_row * num_blocks_col * pruning_rate)
        sorted_scores, _ = torch.sort(block_score)
        threshold = sorted_scores[num_blocks_rm]

        block_score = torch.sum(block_score_sep, dim=(-2, -1)) / (blockdim[0] * blockdim[1])
        block_mask = (block_score >= threshold).float()
        block_mask = block_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(block_score_sep)
        mask = blocks_to_matrix(block_mask, num_blocks_row, num_blocks_col, blockdim[0], blockdim[1])
        if isinstance(module, nn.Conv2d):
            cout, cin, hk, wk = module.weight.data.shape
            mask = mask.reshape(cout, cin, hk, wk)

        module.mask = mask


def global_prune_model(targets: Dict[str, nn.Module], pruning_rate, total_param, block_policy, score_func):
    block_scores = []
    block_dim_map = []
    block_infos = {}
    for name, module in targets.items():
        weight = module.weight.data
        blockdim = get_block_dim(block_policy, name, module)
        if isinstance(module, nn.Conv2d):
            cout, cin, hk, wk = weight.shape
            weight = weight.reshape(cout, cin * hk * wk)

        weight_blocks, num_blocks_row, num_blocks_col = matrix_to_blocks(weight, *blockdim)
        block_score, block_score_sep = score_func(weight_blocks)
        block_scores.append(block_score)
        block_dim_map.append(blockdim[0] * blockdim[1] * torch.ones_like(block_score))
        block_infos[name] = (block_score_sep, num_blocks_row, num_blocks_col, blockdim)

    block_scores = torch.cat(block_scores)
    block_dim_map = torch.cat(block_dim_map)
    num_params_to_rm = int(total_param * pruning_rate)
    sorted_scores, sorted_indices = torch.sort(block_scores)
    param_cum = torch.cumsum(block_dim_map[sorted_indices], dim=0)
    cutoff_index = torch.where((num_params_to_rm < param_cum).float() == 1)[0][0]
    threshold = sorted_scores[cutoff_index]

    total_sparsity = 0
    for name, module in targets.items():
        block_score_sep, num_blocks_row, num_blocks_col, block_dim = block_infos[name]
        block_score = torch.sum(block_score_sep, dim=(-2, -1)) / (block_dim[0] * block_dim[1])
        block_mask = (block_score >= threshold).float()
        block_mask = block_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(block_score_sep)
        mask = blocks_to_matrix(block_mask, num_blocks_row, num_blocks_col, *block_dim)
        if isinstance(module, nn.Conv2d):
            cout, cin, hk, wk = module.weight.data.shape
            mask = mask.reshape(cout, cin, hk, wk)

        layer_sparsity = 1 - mask.sum().item() / mask.numel()
        total_sparsity += module.weight.data.numel() / total_param * layer_sparsity
        module.mask = mask


def mag_score_func(weighted_blocks):
    block_score_sep = torch.sqrt(torch.square(weighted_blocks))
    _, blockrow, blockcol = weighted_blocks.shape
    block_score = torch.sum(block_score_sep, dim=(-2, -1)) / (blockrow * blockcol)
    return block_score, block_score_sep


def darb_adjust_mask_for_model(target_layers: Dict[str, nn.Module], block_sizes: torch.Tensor):
    adjust_mask_funcs = [
        _get_adjusted_row_mask_normal,
        _get_adjusted_row_mask_zero_density
    ]

    for name, target in target_layers.items():  # for each layer
        mask_matrix = extract_mask_tensor_in_matrix_form(target)
        layer_block_sizes = block_sizes[name]
        row_list = []
        """
        block_sizes: 2, 4, inf, 8
        corresponding density: 0.5, 0.25, 0, 0.125

        full weight
        1 2 3 4 5 6 7 8 -- 2
        5 6 7 8 1 2 3 4 -- 4
        7 8 1 2 3 5 6 3 -- inf
        7 8 1 2 3 5 6 3 -- 8
        """
        for row, size in zip(mask_matrix, layer_block_sizes):
            """
            row = 1 2 3 4 5 6 7 8
            size = 2

            partition = 
            1 2
            3 4
            5 6
            7 8

            num_partitions = 4
            nnz_indices_2d = 1, 1, 1, 1
            nnz_indices_1d = [0, 2, 4, 6] + [1, 1, 1, 1] = [1, 3, 5, 7]
            """
            val = adjust_mask_funcs[(size == torch.inf).int().item()](row, size)
            row_list.append(val)

        adjusted_mask = torch.vstack(row_list)
        adjusted_mask = matrix_to_tensor(adjusted_mask, target)
        target.mask = adjusted_mask


def _get_adjusted_row_mask_zero_density(row: torch.Tensor, size: torch.Tensor):
    return torch.zeros(row.numel())


def _get_adjusted_row_mask_normal(row: torch.Tensor, size: torch.Tensor):
    partition = row.reshape(-1, size.int().item())
    num_partitions = partition.shape[0]
    nnz_indices_2d = torch.argmax(np.abs(partition), 1)
    nnz_indices_1d = (torch.arange(num_partitions) * size + nnz_indices_2d).long()
    val = torch.zeros(row.numel())
    val[nnz_indices_1d] = 1
    return val


def extract_mask_tensor_in_matrix_form(target: nn.Module):
    if isinstance(target, nn.Conv2d):
        mask_tensor = target.mask.data
        cout, cin, hk, wk = mask_tensor.shape
        mask_matrix = mask_tensor.reshape(cout, cin * hk * wk)
    else:
        mask_matrix = target.mask.data
    return mask_matrix


def matrix_to_tensor(matrix: torch.Tensor, target: nn.Module):
    if isinstance(target, nn.Conv2d):
        cout, cin, hk, wk = target.weight.data.shape
        ret = matrix.reshape(cout, cin, hk, wk)
    else:
        ret = matrix
    return ret


def darb_get_model_block_sizes(target_layers: Dict[str, nn.Module]):
    ret = {}
    model_density = get_sparsity(target_layers)['density']
    for name, target in target_layers.items():
        mask_matrix = extract_mask_tensor_in_matrix_form(target)
        r, c = mask_matrix.shape
        layer_row_densities = _get_layer_row_density(mask_matrix)
        block_sizes = _get_darb_block_sizes(layer_row_densities, model_density, c)
        ret[name] = block_sizes

    return ret


def _get_layer_row_density(mask_matrix: torch.Tensor):
    r, c = mask_matrix.shape
    blocks = (mask_matrix != 0).float()
    blocks = blocks.sum(1)
    row_densities = blocks / c
    return row_densities


def _get_darb_block_sizes(layer_row_densities: torch.Tensor, model_density: float, ncol: int):
    log_max_block = np.floor(np.log2(ncol))

    cand_block_sizes = np.arange(0, log_max_block)
    cand_block_sizes = np.array([2 ** b for b in cand_block_sizes if is_block_size_candidate_valid(2 ** b, ncol)])
    cand_block_sizes = torch.Tensor(cand_block_sizes)

    round_targets = torch.concat((1 / cand_block_sizes, torch.Tensor([0])))
    round_targets = torch.flip(round_targets, [0])

    rounddown_indices = torch.bucketize(layer_row_densities, round_targets) - 1
    roundup_flag = (layer_row_densities >= model_density).float()
    round_indices = (rounddown_indices + roundup_flag).long()
    rounded_row_densities = round_targets[round_indices]

    """
    Some rows might end up being completely removed 
    e.g. 
    - model density = 0.74
    - row density = 0.03
    - 1/max_possible_block_size = 0.0625
    ret rounded_row_densities = 0
    """

    block_sizes = 1 / rounded_row_densities  # may contain torch.inf
    return block_sizes


DEFAULT_BLOCK_SEARCH_SPACE = {
    'br': np.arange(1, 6000),
    'bc': np.arange(1, 6000),
}


def is_block_size_candidate_valid(candidate: int, dim: int):
    if candidate > dim:
        return False
    if dim % candidate != 0:
        return False
    else:
        return True


def conv_unstructured_single(cout, cin, hk, wk, search_space, tau_acc):
    return [(1, 1)]


def conv_unstructured_channel(cout, cin, hk, wk, search_space, tau_acc):
    return [(1, hk * wk)]


def conv_filter(ccout, cin, hk, wk, search_space, tau_acc):
    return [(1, cin * hk * wk)]


def conv_structured_channel(cout, cin, hk, wk, search_space, tau_acc):
    return [(cout, hk * wk)]


def conv_mixed_structured_channel_filter(cout, cin, hk, wk, search_space, tau_acc):
    flag = np.random.randint(0, 2)
    if flag == 0:
        return conv_structured_channel(cout, cin, hk, wk, search_space, tau_acc)
    else:
        return conv_filter(cout, cin, hk, wk, search_space, tau_acc)


def conv_mixed_unstructured_channel_filter(cout, cin, hk, wk, search_space, tau_acc):
    flag = np.random.randint(0, 2)
    if flag == 0:
        return conv_unstructured_channel(cout, cin, hk, wk, search_space, tau_acc)
    else:
        return conv_filter(cout, cin, hk, wk, search_space, tau_acc)


def conv_mix(cout, cin, hk, wk, search_space, tau_acc):
    return [
        (1, 1),  # unstructured
        (1, hk * wk),  # unstructured channel
        (cout, hk * wk),  # structured channel
        (1, cin * hk * wk),  # filter pruning
        (cout, 1),  # conv column pruning
    ]


def conv_block(cout, cin, hk, wk, search_space, tau_acc):
    possible_brs = np.array([b for b in search_space['br'] if is_block_size_candidate_valid(b, cout)])
    possible_bcs = np.array([b for b in search_space['bc'] if is_block_size_candidate_valid(b, cin * hk * wk)])

    possible_br_bc_set = list(itertools.product(possible_brs, possible_bcs))

    if 0 < tau_acc < 1:
        total_ele = cout * cin * hk * wk
        max_param_one_block = tau_acc * total_ele
    else:
        max_param_one_block = tau_acc

    br_bc_candidates = list(filter(lambda x: x[0] * x[1] <= max_param_one_block, possible_br_bc_set))
    if len(br_bc_candidates) == 0:
        br_bc_candidates = [(1, 1)]
    return br_bc_candidates


def fc_mix(num_row, num_column, search_space, tau_acc):
    return [
        (1, 1),
        (num_row, 1),
        (1, num_column)
    ]


def fc_unstructured_single(num_row, num_column, search_space, tau_acc):
    return [(1, 1)]


def fc_block(num_row, num_column, search_space, tau_acc):
    possible_brs = np.array([b for b in search_space['br'] if is_block_size_candidate_valid(b, num_row)])
    possible_bcs = np.array(np.array([b for b in search_space['bc'] if is_block_size_candidate_valid(b, num_column)]))
    possible_br_bc_set = list(itertools.product(possible_brs, possible_bcs))

    if 0 < tau_acc < 1:
        total_ele = num_row * num_column
        max_param_one_block = tau_acc * total_ele
    else:
        max_param_one_block = tau_acc
    br_bc_candidates = list(filter(lambda x: x[0] * x[1] <= max_param_one_block, possible_br_bc_set))
    if len(br_bc_candidates) == 0:
        br_bc_candidates = [(1, 1)]
    return br_bc_candidates


CONV_PRUNING_FUNC = {
    'unstructured': conv_unstructured_single,
    'unstructured_channel': conv_unstructured_channel,
    'filter_only': conv_filter,
    'structured_channel': conv_structured_channel,
    'mixed_structured_channel_filter': conv_mixed_structured_channel_filter,
    'mixed_unstructured_channel_filter': conv_mixed_unstructured_channel_filter,
    'block': conv_block,
    'existing': conv_mix
}

FC_PRUNING_FUNC = {
    'unstructured': fc_unstructured_single,
    'block': fc_block,
    'existing': fc_mix
}


def get_block_search_space_for_conv(weight: np.array, mode: str = 'default', search_space={}, tau_acc=1):
    cout, cin, hk, wk = weight.shape
    return CONV_PRUNING_FUNC[mode](cout, cin, hk, wk, search_space, tau_acc)


def get_block_search_space_fc(weight: np.array, mode: str = 'default', search_space={}, tau_acc=1):
    num_out_features, num_in_features = weight.shape
    return FC_PRUNING_FUNC[mode](num_out_features, num_in_features, search_space, tau_acc)


def get_block_search_space_single_layer(
        layer: nn.Module,
        conv_mode: str = 'unstructured',
        fc_mode: str = 'unstructured',
        search_space={},
        tau_acc=1
):
    weight = layer.weight.data
    if isinstance(layer, nn.Conv2d):
        return get_block_search_space_for_conv(weight, conv_mode, search_space, tau_acc)
    elif isinstance(layer, nn.Linear):
        return get_block_search_space_fc(weight, fc_mode, search_space, tau_acc)


def get_search_space(usr_valid_brs=(), usr_valid_bcs=()):
    ret = {}
    if len(usr_valid_brs) == 0:
        ret['br'] = DEFAULT_BLOCK_SEARCH_SPACE['br']
    elif len(usr_valid_brs) != 0:
        ret['br'] = set(usr_valid_brs).intersection(set(DEFAULT_BLOCK_SEARCH_SPACE['br'].tolist()))

    if len(usr_valid_bcs) == 0:
        ret['bc'] = DEFAULT_BLOCK_SEARCH_SPACE['bc']
    elif len(usr_valid_brs) != 0:
        ret['bc'] = set(usr_valid_brs).intersection(set(DEFAULT_BLOCK_SEARCH_SPACE['bc'].tolist()))

    return ret


def get_block_search_space_model(
        model: nn.Module,
        conv_mode: str = 'unstructured',
        fc_mode: str = 'unstructured',
        usr_valid_brs=(),
        usr_valid_bcs=(),
        tau_acc=1,
        op_unique=False,
        filter_func=None
):
    ret = {}

    search_space = get_search_space(usr_valid_brs, usr_valid_bcs)

    for idx, (name, module) in enumerate(model.named_modules()):
        if idx == 0:
            continue

        if isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList):
            continue

        break_down_name = name.split('.')
        break_down_name_wo_parallel = [s for s in break_down_name if s != 'module']
        name_wo_parallel = '.'.join(break_down_name_wo_parallel)

        if is_linear_transform_layer(module):
            if op_unique:
                shape = tuple(module.weight.data.shape)
                if ret.get(shape) is None:
                    ret[shape] = get_block_search_space_single_layer(
                        module, conv_mode, fc_mode, search_space, tau_acc
                    )
            else:
                ret[name_wo_parallel] = get_block_search_space_single_layer(
                    module, conv_mode, fc_mode, search_space, tau_acc
                )

    if filter_func is not None:
        for key, vals in ret.items():
            ret[key] = list(filter(filter_func, vals))

    return ret
