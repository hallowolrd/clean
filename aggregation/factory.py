from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aggregation.base import Aggregator, get_aggregation_method
from aggregation.fisher_kfac_expert import FisherKFACExpertAggregator
from aggregation.history_wolf_kfac_score import HistoryWoLFKFACScoreExpertAggregator
from aggregation.sample_weighted import SampleWeightedAggregator
from aggregation.uniform import UniformAggregator


@dataclass
class AggregatorBundle:
    """
    聚合器打包结果。

    non_expert:
        非专家参数聚合器。
        负责 backbone / router / classifier / shared layers 等参数。

    expert:
        专家参数聚合器。
        负责 MoE experts 参数。
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
                fisher_kfac_expert
                history_wolf_kfac_score

        param_group_name:
            当前聚合器负责的参数组。

            当前支持：
                non_expert
                expert
    """

    method = str(method).lower()

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

    if method == "fisher_kfac_expert":
        if param_group_name != "expert":
            raise ValueError("fisher_kfac_expert 只能用于 expert 参数聚合。")

        return FisherKFACExpertAggregator(
            cfg=cfg,
            param_group_name=param_group_name,
        )

    if method == "history_wolf_kfac_score":
        if param_group_name != "expert":
            raise ValueError("history_wolf_kfac_score 只能用于 expert 参数聚合。")

        # 该聚合器是 K-FAC score + WoLF 历史滤波的专家聚合方法。
        # 它只负责 expert 参数；non_expert 仍然应使用 uniform/sample_weighted 等普通聚合器。
        return HistoryWoLFKFACScoreExpertAggregator(
            cfg=cfg,
            param_group_name=param_group_name,
        )

    raise ValueError(
        f"不支持的聚合方法：{method}。"
        "当前支持：uniform, sample_weighted, fisher_kfac_expert, "
        "history_wolf_kfac_score"
    )


def build_aggregators(cfg: Any) -> AggregatorBundle:
    """
    根据配置创建非专家参数聚合器和专家参数聚合器。

    配置格式：

        agg:
          non_expert:
            method: sample_weighted
          expert:
            method: uniform

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