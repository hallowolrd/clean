from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch

from fl.types import AggregationResult, ClientUpdate
from utils.state_dict_ops import (
    apply_weighted_delta,
    check_finite_state_dict,
    normalize_weights,
)


class Aggregator(ABC):
    """
    聚合器基类。

    所有聚合方法都应该继承这个类，例如：
        1. UniformAggregator
        2. SampleWeightedAggregator
        3. 后续 FisherAggregator
        4. 后续 HistoryWolfAggregator

    这个类只规定统一接口，不绑定具体聚合算法。
    """

    def __init__(
        self,
        cfg: Any,
        param_group_name: str,
    ) -> None:
        """
        初始化聚合器。

        参数：
            cfg:
                全局配置对象。

            param_group_name:
                当前聚合器负责的参数组名称。
                例如：
                    non_expert
                    expert
        """
        self.cfg = cfg
        self.param_group_name = param_group_name

    @property
    @abstractmethod
    def method_name(self) -> str:
        """
        当前聚合方法名称。

        子类需要返回：
            uniform
            sample_weighted
            ...
        """
        raise NotImplementedError

    @abstractmethod
    def compute_weights(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> Dict[int, float]:
        """
        计算每个客户端的聚合权重。

        子类只需要实现这个函数。

        例如：
            uniform:
                每个客户端权重相同。

            sample_weighted:
                按客户端样本数加权。
        """
        raise NotImplementedError

    def aggregate(
        self,
        global_state: Mapping[str, torch.Tensor],
        client_updates: Sequence[ClientUpdate],
        param_names: Optional[Iterable[str]] = None,
        base_state: Optional[Mapping[str, torch.Tensor]] = None,
        strict: bool = True,
    ) -> AggregationResult:
        """
        执行参数聚合。

        这是所有普通加权 delta 聚合方法的公共流程：

            1. 检查客户端更新
            2. 计算客户端权重
            3. 归一化客户端权重
            4. 对指定参数组执行加权 delta 聚合
            5. 返回 AggregationResult

        参数：
            global_state:
                本轮聚合前的全局模型参数。

            client_updates:
                本轮参与训练的客户端更新。

            param_names:
                当前聚合器负责聚合的参数名。
                例如：
                    非专家参数名列表
                    专家参数名列表

            base_state:
                聚合结果写入的基础 state_dict。
                如果为 None，则基于 global_state 生成新 state_dict。
                如果先聚合 non_expert，再聚合 expert，可以把上一步结果传进来。

            strict:
                如果为 True，缺少参数或权重时直接报错。
        """
        self._validate_client_updates(client_updates)

        raw_weights = self.compute_weights(client_updates)
        weights = normalize_weights(raw_weights)

        new_state_dict = apply_weighted_delta(
            global_state=global_state,
            client_updates=client_updates,
            weights=weights,
            param_names=param_names,
            base_state=base_state,
            strict=strict,
        )

        check_finite_state_dict(
            state_dict=new_state_dict,
            param_names=param_names,
        )

        diagnostics = self.build_diagnostics(
            client_updates=client_updates,
            weights=weights,
            param_names=param_names,
        )

        return AggregationResult(
            new_state_dict=new_state_dict,
            weights=weights,
            diagnostics=diagnostics,
        )

    def build_diagnostics(
        self,
        client_updates: Sequence[ClientUpdate],
        weights: Mapping[int, float],
        param_names: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """
        构建聚合诊断信息。

        这些信息后面会写入日志，方便确认：
            1. 当前用的是什么聚合方法
            2. 聚合的是 expert 还是 non_expert
            3. 有多少客户端参与
            4. 每个客户端权重是多少
        """
        param_count = None

        if param_names is not None:
            param_count = len(list(param_names))

        return {
            "method": self.method_name,
            "param_group": self.param_group_name,
            "num_clients": len(client_updates),
            "param_count": param_count,
            "weights": {
                int(client_id): float(weight)
                for client_id, weight in weights.items()
            },
        }

    def _validate_client_updates(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> None:
        """
        检查客户端更新是否合法。
        """
        if len(client_updates) == 0:
            raise ValueError("client_updates 不能为空。")

        seen_client_ids = set()

        for update in client_updates:
            if update.client_id in seen_client_ids:
                raise ValueError(f"重复的 client_id：{update.client_id}")

            seen_client_ids.add(update.client_id)

            if update.num_samples <= 0:
                raise ValueError(
                    f"客户端 {update.client_id} 的 num_samples 必须大于 0，"
                    f"当前值：{update.num_samples}"
                )

            if len(update.model_delta) == 0:
                raise ValueError(
                    f"客户端 {update.client_id} 的 model_delta 为空。"
                )


def collect_num_samples(
    client_updates: Sequence[ClientUpdate],
) -> Dict[int, int]:
    """
    收集每个客户端的样本数。

    输出：
        {
            client_id: num_samples
        }
    """
    return {
        int(update.client_id): int(update.num_samples)
        for update in client_updates
    }


def build_uniform_weights(
    client_updates: Sequence[ClientUpdate],
) -> Dict[int, float]:
    """
    构建均匀权重。

    每个客户端权重相同。
    """
    if len(client_updates) == 0:
        raise ValueError("client_updates 不能为空。")

    weight = 1.0 / len(client_updates)

    return {
        int(update.client_id): weight
        for update in client_updates
    }


def build_sample_weights(
    client_updates: Sequence[ClientUpdate],
) -> Dict[int, float]:
    """
    构建按样本数加权的权重。

    注意：
        这里返回的是未必严格归一化前的权重。
        Aggregator.aggregate() 里会统一调用 normalize_weights。
    """
    if len(client_updates) == 0:
        raise ValueError("client_updates 不能为空。")

    weights: Dict[int, float] = {}

    for update in client_updates:
        if update.num_samples <= 0:
            raise ValueError(
                f"客户端 {update.client_id} 的 num_samples 必须大于 0，"
                f"当前值：{update.num_samples}"
            )

        weights[int(update.client_id)] = float(update.num_samples)

    return weights


def get_aggregation_method(cfg: Any, param_group_name: str) -> str:
    """
    从配置中读取指定参数组的聚合方法。

    参数：
        param_group_name:
            non_expert 或 expert

    示例：
        get_aggregation_method(cfg, "non_expert")
        get_aggregation_method(cfg, "expert")
    """
    if param_group_name not in {"non_expert", "expert"}:
        raise ValueError(
            f"不支持的参数组名称：{param_group_name}。"
            "当前支持：non_expert, expert"
        )

    return str(
        cfg.get(
            f"agg.{param_group_name}.method",
            None,
        )
    )