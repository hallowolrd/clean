from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aggregation.base import Aggregator, get_aggregation_method
from aggregation.fisher_history_wolf import FisherHistoryWolfExpertAggregator
from aggregation.fisher_history_wolf_global import FisherHistoryWolfGlobalAggregator
from aggregation.fisher_only import FisherOnlyExpertAggregator
from aggregation.fisher_only_global import FisherOnlyGlobalAggregator
from aggregation.sample_weighted import SampleWeightedAggregator
from aggregation.uniform import UniformAggregator


@dataclass
class AggregatorBundle:
    """
    聚合器打包结果。

    non_expert:
        非专家参数聚合器。

        在 FL+MoE 场景中：
            负责 backbone / router / classifier / shared layers 等参数。

        在 pure-FL 场景中：
            负责整个普通 FL 模型的所有参数。
            例如 resnet18_fedavg 的全部参数都会进入 non_expert 参数组。

    expert:
        专家参数聚合器。

        在 FL+MoE 场景中：
            负责 MoE experts 参数。

        在 pure-FL 场景中：
            expert 参数组为空，server.py 会自动跳过 expert 聚合。
    """

    non_expert: Aggregator
    expert: Aggregator


def build_aggregator(
    cfg: Any,
    method: str,
    param_group_name: str,
) -> Aggregator:
    """
    根据聚合方法名称创建单个聚合器。

    参数：
        cfg:
            全局配置对象。

        method:
            聚合方法名称。

            当前支持：
                uniform
                sample_weighted
                fisher_only
                fisher_history_wolf
                fisher_only_global
                fisher_history_wolf_global

        param_group_name:
            当前聚合器负责的参数组。

            当前支持：
                non_expert
                expert

    方法使用规则：

        uniform:
            可用于 non_expert 或 expert。

        sample_weighted:
            可用于 non_expert 或 expert。

        fisher_only:
            原 FL+MoE expert-wise Fisher-only 聚合器。
            只能用于 expert 参数组。
            读取 update.extra["expert_kfac"]。

        fisher_history_wolf:
            原 FL+MoE expert-wise Fisher-History-WoLF 聚合器。
            只能用于 expert 参数组。
            读取 update.extra["expert_kfac"]。

        fisher_only_global:
            新增 pure-FL client-wise Fisher-only 聚合器。
            只能用于 non_expert 参数组。
            读取 update.extra["global_fisher"]。

        fisher_history_wolf_global:
            新增 pure-FL client-wise Fisher-History-WoLF 聚合器。
            只能用于 non_expert 参数组。
            读取 update.extra["global_fisher"]。
    """
    method = str(method).lower().strip()

    if param_group_name not in {"non_expert", "expert"}:
        raise ValueError(
            f"不支持的参数组名称：{param_group_name}。"
            "当前支持：non_expert, expert"
        )

    if method == "uniform":
        return UniformAggregator(
            cfg=cfg,
            param_group_name=param_group_name,
        )

    if method == "sample_weighted":
        return SampleWeightedAggregator(
            cfg=cfg,
            param_group_name=param_group_name,
        )

    if method == "fisher_only":
        if param_group_name != "expert":
            raise ValueError(
                "fisher_only 是 FL+MoE 的 expert-wise 聚合方法，"
                "只能用于 expert 参数组。"
                "如果你是在 pure-FL 整模型场景使用 Fisher，请改用 fisher_only_global。"
            )

        return FisherOnlyExpertAggregator(
            cfg=cfg,
            param_group_name=param_group_name,
        )

    if method == "fisher_history_wolf":
        if param_group_name != "expert":
            raise ValueError(
                "fisher_history_wolf 是 FL+MoE 的 expert-wise 聚合方法，"
                "只能用于 expert 参数组。"
                "如果你是在 pure-FL 整模型场景使用 Fisher-History-WoLF，"
                "请改用 fisher_history_wolf_global。"
            )

        return FisherHistoryWolfExpertAggregator(
            cfg=cfg,
            param_group_name=param_group_name,
        )

    if method == "fisher_only_global":
        if param_group_name != "non_expert":
            raise ValueError(
                "fisher_only_global 是 pure-FL 的整模型 client-wise 聚合方法，"
                "只能用于 non_expert 参数组。"
                "pure-FL 模型的全部参数应该在 models/param_groups.py 中进入 non_expert。"
            )

        return FisherOnlyGlobalAggregator(
            cfg=cfg,
            param_group_name=param_group_name,
        )

    if method == "fisher_history_wolf_global":
        if param_group_name != "non_expert":
            raise ValueError(
                "fisher_history_wolf_global 是 pure-FL 的整模型 client-wise 聚合方法，"
                "只能用于 non_expert 参数组。"
                "pure-FL 模型的全部参数应该在 models/param_groups.py 中进入 non_expert。"
            )

        return FisherHistoryWolfGlobalAggregator(
            cfg=cfg,
            param_group_name=param_group_name,
        )

    raise ValueError(
        f"不支持的聚合方法：{method}。"
        "当前支持："
        "uniform, sample_weighted, "
        "fisher_only, fisher_history_wolf, "
        "fisher_only_global, fisher_history_wolf_global"
    )


def build_aggregators(cfg: Any) -> AggregatorBundle:
    """
    根据配置创建非专家参数聚合器和专家参数聚合器。

    FL+MoE 配置示例：

        agg:
          non_expert:
            method: sample_weighted
          expert:
            method: fisher_only

    或者：

        agg:
          non_expert:
            method: sample_weighted
          expert:
            method: fisher_history_wolf

    pure-FL 配置示例：

        agg:
          non_expert:
            method: uniform
          expert:
            method: uniform

    或者：

        agg:
          non_expert:
            method: fisher_only_global
          expert:
            method: uniform

    或者：

        agg:
          non_expert:
            method: fisher_history_wolf_global
          expert:
            method: uniform

    注意：
        pure-FL 模型没有 expert 参数。
        这里仍然构建 expert 聚合器，是为了兼容原有 AggregatorBundle 结构。
        server.py 会在 expert 参数组为空时自动跳过 expert 聚合。

    返回：
        AggregatorBundle(
            non_expert=...,
            expert=...,
        )
    """
    non_expert_method = get_aggregation_method(
        cfg=cfg,
        param_group_name="non_expert",
    )
    expert_method = get_aggregation_method(
        cfg=cfg,
        param_group_name="expert",
    )

    non_expert_aggregator = build_aggregator(
        cfg=cfg,
        method=non_expert_method,
        param_group_name="non_expert",
    )

    expert_aggregator = build_aggregator(
        cfg=cfg,
        method=expert_method,
        param_group_name="expert",
    )

    return AggregatorBundle(
        non_expert=non_expert_aggregator,
        expert=expert_aggregator,
    )


def build_non_expert_aggregator(cfg: Any) -> Aggregator:
    """
    只创建非专家参数聚合器。

    一般 server.py 里更推荐直接用 build_aggregators()。
    这个函数主要用于测试或调试。

    non_expert 参数组当前支持：
        uniform
        sample_weighted
        fisher_only_global
        fisher_history_wolf_global

    注意：
        fisher_only 和 fisher_history_wolf 是 expert-wise 方法，
        不允许用于 non_expert 参数组。
    """
    method = get_aggregation_method(
        cfg=cfg,
        param_group_name="non_expert",
    )

    return build_aggregator(
        cfg=cfg,
        method=method,
        param_group_name="non_expert",
    )


def build_expert_aggregator(cfg: Any) -> Aggregator:
    """
    只创建专家参数聚合器。

    一般 server.py 里更推荐直接用 build_aggregators()。
    这个函数主要用于测试或调试。

    expert 参数组当前支持：
        uniform
        sample_weighted
        fisher_only
        fisher_history_wolf

    注意：
        fisher_only_global 和 fisher_history_wolf_global 是 pure-FL 整模型方法，
        不允许用于 expert 参数组。
    """
    method = get_aggregation_method(
        cfg=cfg,
        param_group_name="expert",
    )

    return build_aggregator(
        cfg=cfg,
        method=method,
        param_group_name="expert",
    )