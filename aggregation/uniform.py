from __future__ import annotations

from typing import Dict, Sequence

from aggregation.base import Aggregator, build_uniform_weights
from fl.types import ClientUpdate


class UniformAggregator(Aggregator):
    """
    直接平均聚合器。

    权重规则：
        每个参与客户端权重相同。

    公式：
        w_i = 1 / K

    其中：
        K 是本轮参与聚合的客户端数量。

    聚合公式：
        theta_new = theta_global + sum_i w_i * delta_i
    """

    @property
    def method_name(self) -> str:
        """返回当前聚合方法名称。"""
        return "uniform"

    def compute_weights(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> Dict[int, float]:
        """
        计算直接平均权重。

        输入：
            client_updates:
                本轮参与训练的客户端更新。

        输出：
            {
                client_id: weight
            }

        示例：
            如果本轮有 4 个客户端参与：
                client 0 -> 0.25
                client 1 -> 0.25
                client 2 -> 0.25
                client 3 -> 0.25
        """
        return build_uniform_weights(client_updates)