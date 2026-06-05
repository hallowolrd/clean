from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import torch


TensorDict = Dict[str, torch.Tensor]


@dataclass
class ClientUpdate:
    """
    客户端每一轮训练后上传给服务端的结果。

    这个结构是 client 和 server 之间的统一接口。
    后面无论是 FedAvg、ExpertFedAvg、Fisher、history filter、Bayes，
    都尽量往这个结构里扩展，而不是让 client.py 和 server.py 互相强耦合。
    """

    # 客户端编号
    client_id: int

    # 当前联邦训练轮数
    round_id: int

    # 当前客户端本地训练样本数
    num_samples: int

    # 本地模型相对全局模型的参数变化量
    # 公式：
    #   model_delta = local_model - global_model
    model_delta: TensorDict

    # 客户端本地训练指标
    # 例如：
    #   train_loss
    #   train_acc
    metrics: Dict[str, float] = field(default_factory=dict)

    # 预留扩展字段
    # 后面可以放：
    #   expert_usage
    #   fisher_diag
    #   sgld_mean
    #   sgld_var
    #   router_stats
    extra: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """
        返回适合写日志的轻量摘要。

        注意：
            不包含 model_delta，因为 tensor 太大，不适合直接写入日志。
        """
        return {
            "client_id": self.client_id,
            "round_id": self.round_id,
            "num_samples": self.num_samples,
            "metrics": dict(self.metrics),
            "extra_keys": sorted(self.extra.keys()),
        }


@dataclass
class AggregationResult:
    """
    聚合器返回给 server 的结果。

    所有聚合方法都应该返回这个结构。
    这样 server.py 不需要关心当前到底是 uniform 还是 sample_weighted。
    """

    # 聚合后的新全局模型参数
    new_state_dict: TensorDict

    # 每个客户端的最终聚合权重
    # 例如：
    #   {0: 0.1, 1: 0.2, 2: 0.7}
    weights: Dict[int, float]

    # 聚合诊断信息
    # 例如：
    #   method
    #   num_clients
    #   total_samples
    #   param_group
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """
        返回适合写日志的轻量摘要。

        注意：
            不包含 new_state_dict，因为模型参数太大。
        """
        return {
            "weights": dict(self.weights),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass
class RoundResult:
    """
    每一轮联邦训练结束后的结果摘要。

    这个结构主要用于日志、results.csv、后续实验分析。
    """

    # 当前轮数
    round_id: int

    # 本轮参与训练的客户端编号
    selected_clients: List[int]

    # 测试集 loss
    test_loss: float

    # 测试集准确率
    test_acc: float

    # 当前历史最佳准确率
    best_acc: float

    # 本轮客户端训练指标摘要
    client_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)

    # 本轮聚合信息摘要
    aggregation_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        转成普通 dict，方便写 json / csv。
        """
        return {
            "round_id": self.round_id,
            "selected_clients": list(self.selected_clients),
            "test_loss": float(self.test_loss),
            "test_acc": float(self.test_acc),
            "best_acc": float(self.best_acc),
            "client_metrics": self.client_metrics,
            "aggregation_info": self.aggregation_info,
        }


@dataclass
class TrainState:
    """
    服务端训练状态。

    用于 checkpoint 保存和断点续训。
    第一版可以先只用其中一部分字段。
    """

    # 当前已经完成的轮数
    round_id: int = 0

    # 当前历史最佳准确率
    best_acc: float = 0.0

    # 最佳模型出现在哪一轮
    best_round: int = 0

    # 额外状态
    # 后面可以放：
    #   history filter state
    #   bayes prior state
    #   fisher running state
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        转成普通 dict，方便保存 checkpoint。
        """
        return {
            "round_id": self.round_id,
            "best_acc": self.best_acc,
            "best_round": self.best_round,
            "extra": self.extra,
        }


def average_client_metric(
    client_updates: List[ClientUpdate],
    metric_name: str,
    weighted: bool = True,
    default: Optional[float] = None,
) -> Optional[float]:
    """
    统计客户端指标平均值。

    参数：
        client_updates:
            本轮客户端上传结果。

        metric_name:
            指标名，例如 train_loss / train_acc。

        weighted:
            是否按客户端样本数加权。

        default:
            如果没有任何客户端包含该指标，则返回 default。
    """
    values = []

    for update in client_updates:
        if metric_name not in update.metrics:
            continue

        value = float(update.metrics[metric_name])
        weight = int(update.num_samples) if weighted else 1
        values.append((value, weight))

    if len(values) == 0:
        return default

    total_weight = sum(weight for _, weight in values)

    if total_weight <= 0:
        return default

    return sum(value * weight for value, weight in values) / total_weight


def collect_client_metrics(
    client_updates: List[ClientUpdate],
) -> Dict[int, Dict[str, float]]:
    """
    把客户端指标整理成 dict。

    输出格式：
        {
            client_id: {
                "train_loss": ...,
                "train_acc": ...
            }
        }
    """
    return {
        update.client_id: dict(update.metrics)
        for update in client_updates
    }


def get_update_client_id(update: ClientUpdate | Mapping[str, Any]) -> int:
    """
    兼容 ClientUpdate 和 dict，读取 client_id。
    """
    if isinstance(update, Mapping):
        return int(update["client_id"])

    return int(update.client_id)


def get_update_num_samples(update: ClientUpdate | Mapping[str, Any]) -> int:
    """
    兼容 ClientUpdate 和 dict，读取 num_samples。
    """
    if isinstance(update, Mapping):
        return int(update["num_samples"])

    return int(update.num_samples)


def get_update_model_delta(
    update: ClientUpdate | Mapping[str, Any],
) -> Mapping[str, torch.Tensor]:
    """
    兼容 ClientUpdate 和 dict，读取 model_delta。
    """
    if isinstance(update, Mapping):
        return update["model_delta"]

    return update.model_delta