from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch


StateDict = Mapping[str, torch.Tensor]
MutableStateDict = Dict[str, torch.Tensor]


def clone_state_dict(state_dict: StateDict) -> MutableStateDict:
    """
    深拷贝 state_dict。

    用途：
        1. 保存全局模型参数快照
        2. 避免后续原地修改污染原始模型
    """
    return {
        name: tensor.detach().clone()
        for name, tensor in state_dict.items()
    }


def detach_state_dict(state_dict: StateDict) -> MutableStateDict:
    """
    断开 state_dict 中 tensor 的计算图。

    注意：
        这里只 detach，不 clone。
        如果需要完全独立副本，请使用 clone_state_dict。
    """
    return {
        name: tensor.detach()
        for name, tensor in state_dict.items()
    }


def state_dict_to(
    state_dict: StateDict,
    device: torch.device | str,
) -> MutableStateDict:
    """
    把 state_dict 移动到指定设备。

    示例：
        state_dict_to(state, "cpu")
        state_dict_to(state, "cuda")
    """
    return {
        name: tensor.to(device)
        for name, tensor in state_dict.items()
    }


def subtract_state_dict(
    local_state: StateDict,
    global_state: StateDict,
    param_names: Optional[Iterable[str]] = None,
    strict: bool = True,
) -> MutableStateDict:
    """
    计算客户端本地模型相对全局模型的参数变化量。

    公式：
        delta = local_state - global_state

    说明：
        1. 只处理浮点 tensor
        2. 非浮点 tensor 会跳过
        3. 例如 BatchNorm 的 num_batches_tracked 通常是 int64，不适合做加减
    """
    names = _resolve_param_names(
        state_dict=global_state,
        param_names=param_names,
    )

    delta: MutableStateDict = {}

    for name in names:
        if name not in local_state:
            if strict:
                raise KeyError(f"local_state 缺少参数：{name}")
            continue

        if name not in global_state:
            if strict:
                raise KeyError(f"global_state 缺少参数：{name}")
            continue

        local_tensor = local_state[name]
        global_tensor = global_state[name]

        if not _is_float_tensor(local_tensor):
            continue

        if not _is_float_tensor(global_tensor):
            continue

        delta[name] = local_tensor.detach() - global_tensor.detach()

    return delta


def apply_delta(
    base_state: StateDict,
    delta: StateDict,
    param_names: Optional[Iterable[str]] = None,
    strict: bool = False,
) -> MutableStateDict:
    """
    把 delta 加到 base_state 上。

    公式：
        new_state = base_state + delta

    用途：
        适合单个客户端更新、调试或特殊聚合。
    """
    new_state = clone_state_dict(base_state)

    names = _resolve_param_names(
        state_dict=base_state,
        param_names=param_names,
    )

    for name in names:
        if name not in delta:
            if strict:
                raise KeyError(f"delta 缺少参数：{name}")
            continue

        if not _is_float_tensor(new_state[name]):
            continue

        new_state[name] = new_state[name] + delta[name].to(new_state[name].device)

    return new_state


def apply_weighted_delta(
    global_state: StateDict,
    client_updates: Sequence[Any],
    weights: Mapping[int, float],
    param_names: Optional[Iterable[str]] = None,
    base_state: Optional[StateDict] = None,
    strict: bool = True,
) -> MutableStateDict:
    """
    对多个客户端 delta 做加权聚合，并更新到全局模型上。

    公式：
        delta_global = sum_i weight_i * delta_i
        new_state = global_state + delta_global

    参数：
        global_state:
            本轮聚合前的全局模型参数。

        client_updates:
            客户端上传结果列表。
            每个 update 需要包含：
                update.client_id
                update.model_delta

            也兼容 dict 形式：
                update["client_id"]
                update["model_delta"]

        weights:
            每个客户端的聚合权重。
            例如：
                {0: 0.2, 1: 0.3, 2: 0.5}

        param_names:
            只聚合指定参数。
            后面 FL + MoE 解耦时会用它区分：
                非专家参数
                专家参数

        base_state:
            聚合结果写入的基础 state_dict。
            如果为 None，就从 global_state clone 一份。
            如果先聚合非专家参数，再聚合专家参数，可以把上一步结果传进来。

        strict:
            如果为 True，缺少权重或缺少 delta 时直接报错。
            如果为 False，则跳过缺失项。
    """
    if len(client_updates) == 0:
        raise ValueError("client_updates 不能为空。")

    _validate_weights(weights)

    if base_state is None:
        new_state = clone_state_dict(global_state)
    else:
        new_state = clone_state_dict(base_state)

    names = _resolve_param_names(
        state_dict=global_state,
        param_names=param_names,
    )

    for name in names:
        global_tensor = global_state[name]

        # 非浮点 tensor 不参与聚合，保留 base_state 中原来的值。
        if not _is_float_tensor(global_tensor):
            continue

        total_delta = torch.zeros_like(global_tensor)

        for update in client_updates:
            client_id = _get_client_id(update)
            model_delta = _get_model_delta(update)

            if client_id not in weights:
                if strict:
                    raise KeyError(f"weights 缺少客户端 {client_id} 的权重")
                continue

            if name not in model_delta:
                if strict:
                    raise KeyError(
                        f"客户端 {client_id} 的 model_delta 缺少参数：{name}"
                    )
                continue

            weight = float(weights[client_id])
            delta_tensor = model_delta[name].to(global_tensor.device)

            total_delta = total_delta + weight * delta_tensor

        new_state[name] = global_tensor + total_delta

    return new_state


def weighted_average_state_dicts(
    state_dicts: Sequence[StateDict],
    weights: Sequence[float],
    param_names: Optional[Iterable[str]] = None,
    strict: bool = True,
) -> MutableStateDict:
    """
    对多个 state_dict 直接做加权平均。

    公式：
        avg_state = sum_i weight_i * state_i

    注意：
        联邦学习里更推荐使用：
            global_state + 加权 delta

        这个函数主要用于诊断或特殊聚合。
    """
    if len(state_dicts) == 0:
        raise ValueError("state_dicts 不能为空。")

    if len(state_dicts) != len(weights):
        raise ValueError(
            f"state_dicts 和 weights 数量不一致："
            f"{len(state_dicts)} vs {len(weights)}"
        )

    normalized_weights = normalize_weight_list(weights)

    reference_state = state_dicts[0]
    names = _resolve_param_names(
        state_dict=reference_state,
        param_names=param_names,
    )

    result = clone_state_dict(reference_state)

    for name in names:
        reference_tensor = reference_state[name]

        if not _is_float_tensor(reference_tensor):
            continue

        avg_tensor = torch.zeros_like(reference_tensor)

        for state, weight in zip(state_dicts, normalized_weights):
            if name not in state:
                if strict:
                    raise KeyError(f"某个 state_dict 缺少参数：{name}")
                continue

            if not _is_float_tensor(state[name]):
                continue

            avg_tensor = avg_tensor + float(weight) * state[name].to(reference_tensor.device)

        result[name] = avg_tensor

    return result


def scale_state_dict(
    state_dict: StateDict,
    scale: float,
    param_names: Optional[Iterable[str]] = None,
) -> MutableStateDict:
    """
    对 state_dict 中的浮点 tensor 乘一个系数。
    """
    result = clone_state_dict(state_dict)

    names = _resolve_param_names(
        state_dict=state_dict,
        param_names=param_names,
    )

    for name in names:
        if _is_float_tensor(result[name]):
            result[name] = result[name] * float(scale)

    return result


def add_state_dicts(
    state_a: StateDict,
    state_b: StateDict,
    param_names: Optional[Iterable[str]] = None,
    strict: bool = True,
) -> MutableStateDict:
    """
    两个 state_dict 相加。

    只处理浮点 tensor。
    非浮点 tensor 保留 state_a 的值。
    """
    result = clone_state_dict(state_a)

    names = _resolve_param_names(
        state_dict=state_a,
        param_names=param_names,
    )

    for name in names:
        if name not in state_b:
            if strict:
                raise KeyError(f"state_b 缺少参数：{name}")
            continue

        if _is_float_tensor(result[name]) and _is_float_tensor(state_b[name]):
            result[name] = result[name] + state_b[name].to(result[name].device)

    return result


def subtract_state_dicts(
    state_a: StateDict,
    state_b: StateDict,
    param_names: Optional[Iterable[str]] = None,
    strict: bool = True,
) -> MutableStateDict:
    """
    两个 state_dict 相减。

    公式：
        result = state_a - state_b

    这个函数和 subtract_state_dict 作用类似，
    保留它是为了让命名在通用场景下更直观。
    """
    return subtract_state_dict(
        local_state=state_a,
        global_state=state_b,
        param_names=param_names,
        strict=strict,
    )


def state_dict_l2_norm(
    state_dict: StateDict,
    param_names: Optional[Iterable[str]] = None,
) -> float:
    """
    计算 state_dict 中所有浮点 tensor 的整体 L2 norm。

    用途：
        1. 诊断客户端更新幅度
        2. 诊断专家参数变化大小
        3. 后面 history filter 也可能会用到
    """
    names = _resolve_param_names(
        state_dict=state_dict,
        param_names=param_names,
    )

    total = 0.0

    for name in names:
        tensor = state_dict[name]

        if not _is_float_tensor(tensor):
            continue

        value = tensor.detach().float().pow(2).sum().item()
        total += value

    return math.sqrt(total)


def state_dict_cosine_similarity(
    state_a: StateDict,
    state_b: StateDict,
    param_names: Optional[Iterable[str]] = None,
    eps: float = 1e-12,
) -> float:
    """
    计算两个 state_dict 的余弦相似度。

    用途：
        后面做方向诊断、history filter 时会用到。
    """
    names = _resolve_param_names(
        state_dict=state_a,
        param_names=param_names,
    )

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for name in names:
        if name not in state_b:
            continue

        tensor_a = state_a[name]
        tensor_b = state_b[name]

        if not _is_float_tensor(tensor_a):
            continue

        if not _is_float_tensor(tensor_b):
            continue

        a = tensor_a.detach().float().reshape(-1)
        b = tensor_b.detach().float().reshape(-1)

        dot += torch.dot(a, b).item()
        norm_a += torch.dot(a, a).item()
        norm_b += torch.dot(b, b).item()

    if norm_a <= eps or norm_b <= eps:
        return 0.0

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b) + eps)


def check_finite_state_dict(
    state_dict: StateDict,
    param_names: Optional[Iterable[str]] = None,
) -> None:
    """
    检查 state_dict 中是否存在 NaN 或 Inf。

    如果发现异常，直接抛出 ValueError。
    """
    names = _resolve_param_names(
        state_dict=state_dict,
        param_names=param_names,
    )

    for name in names:
        tensor = state_dict[name]

        if not _is_float_tensor(tensor):
            continue

        if not torch.isfinite(tensor).all():
            raise ValueError(f"参数 {name} 中存在 NaN 或 Inf。")


def normalize_weights(weights: Mapping[int, float]) -> Dict[int, float]:
    """
    把客户端权重归一化到和为 1。

    输入：
        {0: 10, 1: 20, 2: 30}

    输出：
        {0: 1/6, 1: 2/6, 2: 3/6}
    """
    if len(weights) == 0:
        raise ValueError("weights 不能为空。")

    total = 0.0

    for client_id, weight in weights.items():
        weight = float(weight)

        if not math.isfinite(weight):
            raise ValueError(f"客户端 {client_id} 的权重不是有限数：{weight}")

        if weight < 0:
            raise ValueError(f"客户端 {client_id} 的权重小于 0：{weight}")

        total += weight

    if total <= 0:
        raise ValueError(f"weights 总和必须大于 0，当前总和：{total}")

    return {
        int(client_id): float(weight) / total
        for client_id, weight in weights.items()
    }


def normalize_weight_list(weights: Sequence[float]) -> List[float]:
    """
    把权重列表归一化到和为 1。
    """
    if len(weights) == 0:
        raise ValueError("weights 不能为空。")

    total = 0.0

    for weight in weights:
        weight = float(weight)

        if not math.isfinite(weight):
            raise ValueError(f"存在非有限权重：{weight}")

        if weight < 0:
            raise ValueError(f"存在负权重：{weight}")

        total += weight

    if total <= 0:
        raise ValueError(f"weights 总和必须大于 0，当前总和：{total}")

    return [
        float(weight) / total
        for weight in weights
    ]


def select_state_dict(
    state_dict: StateDict,
    param_names: Iterable[str],
    strict: bool = True,
) -> MutableStateDict:
    """
    从 state_dict 中选出一部分参数。

    用途：
        后面可以用于单独查看专家参数或非专家参数。
    """
    selected: MutableStateDict = {}

    for name in param_names:
        if name not in state_dict:
            if strict:
                raise KeyError(f"state_dict 中不存在参数：{name}")
            continue

        selected[name] = state_dict[name].detach().clone()

    return selected


def _resolve_param_names(
    state_dict: StateDict,
    param_names: Optional[Iterable[str]],
) -> List[str]:
    """
    解析需要处理的参数名列表。

    如果 param_names 为 None，则使用 state_dict 的所有 key。
    """
    if param_names is None:
        return list(state_dict.keys())

    names = list(param_names)

    for name in names:
        if name not in state_dict:
            raise KeyError(f"state_dict 中不存在参数：{name}")

    return names


def _is_float_tensor(tensor: torch.Tensor) -> bool:
    """
    判断 tensor 是否是浮点 tensor。
    """
    return torch.is_tensor(tensor) and torch.is_floating_point(tensor)


def _validate_weights(weights: Mapping[int, float]) -> None:
    """
    检查客户端权重是否合法。

    注意：
        这里只检查非空、有限、非负。
        不强制要求权重和等于 1。
        如果需要归一化，请先调用 normalize_weights。
    """
    if len(weights) == 0:
        raise ValueError("weights 不能为空。")

    for client_id, weight in weights.items():
        weight = float(weight)

        if not math.isfinite(weight):
            raise ValueError(f"客户端 {client_id} 的权重不是有限数：{weight}")

        if weight < 0:
            raise ValueError(f"客户端 {client_id} 的权重小于 0：{weight}")


def _get_client_id(update: Any) -> int:
    """
    从 ClientUpdate 或 dict 中读取 client_id。
    """
    if isinstance(update, Mapping):
        return int(update["client_id"])

    return int(update.client_id)


def _get_model_delta(update: Any) -> Mapping[str, torch.Tensor]:
    """
    从 ClientUpdate 或 dict 中读取 model_delta。
    """
    if isinstance(update, Mapping):
        return update["model_delta"]

    return update.model_delta