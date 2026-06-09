from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch


StateDict = Mapping[str, torch.Tensor]


_EXPERT_ID_PATTERN = re.compile(r"(?:^|\.)experts\.(\d+)(?:\.|$)")


@dataclass(frozen=True)
class ParamGroups:
    """
    模型参数分组结果。

    all:
        state_dict 里的全部参数名和 buffer 名。

    non_expert:
        非专家参数名。
        例如 backbone / router / classifier / BatchNorm buffer 等。

        对 pure-FL 模型，例如 resnet18_fedavg：
            non_expert = all

    expert:
        所有 expert 参数名。

        对 pure-FL 模型：
            expert = []

    expert_by_id:
        每个 expert 单独对应的参数名。
        例如：
            {
                0: ["moe.experts.0.fc.weight", ...],
                1: ["moe.experts.1.fc.weight", ...],
            }

        对 pure-FL 模型：
            expert_by_id = {}
    """

    all: List[str]
    non_expert: List[str]
    expert: List[str]
    expert_by_id: Dict[int, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        转成普通 dict，方便写日志或调试。
        """
        return {
            "all": list(self.all),
            "non_expert": list(self.non_expert),
            "expert": list(self.expert),
            "expert_by_id": {
                int(expert_id): list(names)
                for expert_id, names in self.expert_by_id.items()
            },
        }

    def summary(self) -> Dict[str, Any]:
        """
        返回轻量摘要，不包含完整参数名列表。
        """
        return {
            "num_all": len(self.all),
            "num_non_expert": len(self.non_expert),
            "num_expert": len(self.expert),
            "num_experts_found": len(self.expert_by_id),
            "num_params_by_expert": {
                int(expert_id): len(names)
                for expert_id, names in self.expert_by_id.items()
            },
            "is_pure_fl": bool(len(self.expert) == 0),
        }


def build_param_groups(
    model: torch.nn.Module,
    expected_num_experts: int | None = None,
    strict: bool = True,
) -> ParamGroups:
    """
    根据模型 state_dict 构建参数分组。

    注意：
        这里使用 model.state_dict().keys()，
        而不是 model.named_parameters()。

    原因：
        state_dict 里不仅有可训练参数，
        还有 BatchNorm running_mean / running_var 等 buffer。
        联邦聚合时通常也需要处理这些浮点 buffer。

    pure-FL 兼容：
        对 resnet18_fedavg 这种没有 experts.* 参数的模型，
        会自动把全部 state_dict key 放入 non_expert，
        expert 参数组为空。
    """
    state_dict = model.state_dict()

    return build_param_groups_from_state_dict(
        state_dict=state_dict,
        expected_num_experts=expected_num_experts,
        strict=strict,
    )


def build_param_groups_from_state_dict(
    state_dict: StateDict,
    expected_num_experts: int | None = None,
    strict: bool = True,
) -> ParamGroups:
    """
    根据 state_dict 构建参数分组。

    专家参数识别规则：
        参数名中出现 experts.{id}，
        就认为它属于 expert。

    例如：
        moe.experts.0.fc1.weight -> expert 0
        head.experts.3.bias -> expert 3
        backbone.conv.weight -> non_expert

    pure-FL 模型：
        如果整个 state_dict 里没有任何 experts.{id}，
        说明这是普通 FL 模型或非 MoE 模型。
        此时：
            non_expert = all
            expert = []
            expert_by_id = {}
    """
    all_names = list(state_dict.keys())

    non_expert_names: List[str] = []
    expert_names: List[str] = []
    expert_by_id: Dict[int, List[str]] = {}

    for name in all_names:
        expert_id = get_expert_id_from_name(name)

        if expert_id is None:
            non_expert_names.append(name)
        else:
            expert_names.append(name)
            expert_by_id.setdefault(expert_id, []).append(name)

    groups = ParamGroups(
        all=all_names,
        non_expert=non_expert_names,
        expert=expert_names,
        expert_by_id={
            expert_id: names
            for expert_id, names in sorted(expert_by_id.items())
        },
    )

    validate_param_groups(
        groups=groups,
        state_dict=state_dict,
        expected_num_experts=expected_num_experts,
        strict=strict,
    )

    return groups


def get_expert_id_from_name(name: str) -> int | None:
    """
    从参数名中解析 expert id。

    匹配规则：
        experts.{id}

    示例：
        "moe.experts.0.fc.weight" -> 0
        "experts.3.bias" -> 3
        "backbone.conv.weight" -> None
        "client_side.layer1.0.weight" -> None
        "server_side.fc.weight" -> None
    """
    match = _EXPERT_ID_PATTERN.search(name)

    if match is None:
        return None

    return int(match.group(1))


def is_expert_param_name(name: str) -> bool:
    """
    判断一个参数名是否属于 expert。
    """
    return get_expert_id_from_name(name) is not None


def is_non_expert_param_name(name: str) -> bool:
    """
    判断一个参数名是否属于非 expert。
    """
    return not is_expert_param_name(name)


def is_pure_fl_param_groups(groups: ParamGroups) -> bool:
    """
    判断当前参数分组是否对应 pure-FL / 非 MoE 模型。

    条件：
        没有任何 expert 参数。

    用途：
        server.py 可以用这个函数判断是否需要跳过 expert 聚合。
    """
    return len(groups.expert) == 0 and len(groups.expert_by_id) == 0


def get_param_names(
    groups: ParamGroups,
    param_group_name: str,
) -> List[str]:
    """
    根据参数组名称获取参数名列表。

    支持：
        all
        non_expert
        expert

    示例：
        get_param_names(groups, "non_expert")
        get_param_names(groups, "expert")

    pure-FL 场景：
        get_param_names(groups, "non_expert") 返回全部参数。
        get_param_names(groups, "expert") 返回空列表。
    """
    if param_group_name == "all":
        return list(groups.all)

    if param_group_name == "non_expert":
        return list(groups.non_expert)

    if param_group_name == "expert":
        return list(groups.expert)

    raise ValueError(
        f"不支持的参数组名称：{param_group_name}。"
        "当前支持：all, non_expert, expert"
    )


def get_expert_param_names(
    groups: ParamGroups,
    expert_id: int,
) -> List[str]:
    """
    获取某个 expert 对应的参数名列表。

    pure-FL 场景：
        expert_by_id 为空，因此会返回空列表。
    """
    expert_id = int(expert_id)
    return list(groups.expert_by_id.get(expert_id, []))


def validate_param_groups(
    groups: ParamGroups,
    state_dict: StateDict,
    expected_num_experts: int | None = None,
    strict: bool = True,
) -> None:
    """
    检查参数分组是否合法。

    检查内容：
        1. all 是否覆盖 state_dict 所有 key；
        2. non_expert 和 expert 是否有重叠；
        3. non_expert + expert 是否刚好覆盖 all；
        4. expert_by_id 是否和 expert 一致；
        5. 如果是 MoE 模型，检查 expert id 是否符合 expected_num_experts；
        6. 如果是 pure-FL 模型，允许 expert 为空。

    重要说明：
        原始版本在 strict=True 且 expert 为空时会直接报错。
        这对 MoE 模型是合理的，但对 resnet18_fedavg 这种 pure-FL 模型不合理。

        现在的逻辑是：
            - 如果找到了 expert 参数，继续执行 MoE 严格检查；
            - 如果没有找到 expert 参数，则认为是 pure-FL / 非 MoE 模型，
              允许 expert 为空。
    """
    state_names = set(state_dict.keys())
    all_names = set(groups.all)
    non_expert_names = set(groups.non_expert)
    expert_names = set(groups.expert)

    if all_names != state_names:
        missing = sorted(state_names - all_names)
        extra = sorted(all_names - state_names)
        raise ValueError(
            "ParamGroups.all 和 state_dict keys 不一致。"
            f" missing={missing[:10]}, extra={extra[:10]}"
        )

    overlap = non_expert_names & expert_names
    if overlap:
        raise ValueError(
            "non_expert 和 expert 参数组存在重叠："
            f"{sorted(overlap)[:10]}"
        )

    merged = non_expert_names | expert_names
    if merged != all_names:
        missing = sorted(all_names - merged)
        extra = sorted(merged - all_names)
        raise ValueError(
            "non_expert + expert 没有刚好覆盖 all。"
            f" missing={missing[:10]}, extra={extra[:10]}"
        )

    expert_by_id_names = set()

    for expert_id, names in groups.expert_by_id.items():
        if expert_id < 0:
            raise ValueError(f"expert_id 不能小于 0，当前值：{expert_id}")

        for name in names:
            parsed_expert_id = get_expert_id_from_name(name)

            if parsed_expert_id != expert_id:
                raise ValueError(
                    f"expert_by_id 分组错误：参数 {name} 被放到 expert {expert_id}，"
                    f"但解析结果是 {parsed_expert_id}。"
                )

            expert_by_id_names.add(name)

    if expert_by_id_names != expert_names:
        missing = sorted(expert_names - expert_by_id_names)
        extra = sorted(expert_by_id_names - expert_names)
        raise ValueError(
            "expert_by_id 和 expert 参数组不一致。"
            f" missing={missing[:10]}, extra={extra[:10]}"
        )

    # pure-FL / 非 MoE 模型：
    # 没有任何 experts.{id} 参数时，允许 expert 为空。
    # 这正是 resnet18_fedavg 的情况。
    if len(expert_names) == 0:
        return

    # MoE 模型：
    # 如果已经找到了 expert 参数，则继续保持原来的严格检查。
    if strict and expected_num_experts is not None:
        expected_num_experts = int(expected_num_experts)

        if expected_num_experts <= 0:
            raise ValueError(
                f"expected_num_experts 必须大于 0，当前值：{expected_num_experts}"
            )

        found_expert_ids = sorted(groups.expert_by_id.keys())
        expected_expert_ids = list(range(expected_num_experts))

        if found_expert_ids != expected_expert_ids:
            raise ValueError(
                "模型中找到的 expert id 和期望不一致。"
                f" found={found_expert_ids}, expected={expected_expert_ids}"
            )


def count_tensors_by_group(groups: ParamGroups) -> Dict[str, Any]:
    """
    统计每个参数组包含多少个 tensor。
    """
    return {
        "all": len(groups.all),
        "non_expert": len(groups.non_expert),
        "expert": len(groups.expert),
        "expert_by_id": {
            int(expert_id): len(names)
            for expert_id, names in groups.expert_by_id.items()
        },
    }


def count_numel_by_group(
    state_dict: StateDict,
    groups: ParamGroups,
    only_floating: bool = True,
) -> Dict[str, Any]:
    """
    统计每个参数组包含多少个元素。

    参数：
        only_floating:
            如果为 True，只统计浮点 tensor。
            如果为 False，所有 tensor 都统计。
    """
    return {
        "all": _count_numel(
            state_dict=state_dict,
            names=groups.all,
            only_floating=only_floating,
        ),
        "non_expert": _count_numel(
            state_dict=state_dict,
            names=groups.non_expert,
            only_floating=only_floating,
        ),
        "expert": _count_numel(
            state_dict=state_dict,
            names=groups.expert,
            only_floating=only_floating,
        ),
        "expert_by_id": {
            int(expert_id): _count_numel(
                state_dict=state_dict,
                names=names,
                only_floating=only_floating,
            )
            for expert_id, names in groups.expert_by_id.items()
        },
    }


def summarize_param_groups(
    state_dict: StateDict,
    groups: ParamGroups,
) -> Dict[str, Any]:
    """
    汇总参数分组信息，方便打印日志。

    输出包括：
        1. tensor 数量；
        2. 浮点元素数量；
        3. 所有元素数量；
        4. 是否是 pure-FL / 非 MoE 模型。
    """
    return {
        "tensor_counts": count_tensors_by_group(groups),
        "floating_numel": count_numel_by_group(
            state_dict=state_dict,
            groups=groups,
            only_floating=True,
        ),
        "all_numel": count_numel_by_group(
            state_dict=state_dict,
            groups=groups,
            only_floating=False,
        ),
        "is_pure_fl": bool(is_pure_fl_param_groups(groups)),
    }


def filter_names_by_prefix(
    names: Iterable[str],
    prefixes: Sequence[str],
) -> List[str]:
    """
    按前缀筛选参数名。

    这个函数不是主流程必须的，主要用于调试。
    """
    prefixes = tuple(prefixes)

    return [
        name
        for name in names
        if name.startswith(prefixes)
    ]


def filter_names_by_keyword(
    names: Iterable[str],
    keywords: Sequence[str],
) -> List[str]:
    """
    按关键词筛选参数名。

    这个函数不是主流程必须的，主要用于调试。
    """
    return [
        name
        for name in names
        if any(keyword in name for keyword in keywords)
    ]


def _count_numel(
    state_dict: StateDict,
    names: Iterable[str],
    only_floating: bool,
) -> int:
    """
    统计指定参数名对应 tensor 的元素数量。
    """
    total = 0

    for name in names:
        if name not in state_dict:
            raise KeyError(f"state_dict 中不存在参数：{name}")

        tensor = state_dict[name]

        if only_floating and not torch.is_floating_point(tensor):
            continue

        total += int(tensor.numel())

    return total