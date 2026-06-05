from __future__ import annotations

from typing import Dict, Sequence

from aggregation.base import Aggregator, build_sample_weights
from fl.types import ClientUpdate


class SampleWeightedAggregator(Aggregator):
    """
    按样本数加权聚合器。

    权重规则：
        客户端样本数越多，聚合权重越大。

    原始权重：
        raw_w_i = n_i

    归一化后：
        w_i = n_i / sum_j n_j

    其中：
        n_i 是客户端 i 的本地训练样本数。

    聚合公式：
        theta_new = theta_global + sum_i w_i * delta_i
    """

    @property
    def method_name(self) -> str:
        """返回当前聚合方法名称。"""
        return "sample_weighted"

    def compute_weights(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> Dict[int, float]:
        """
        计算按样本数加权的客户端权重。

        输入：
            client_updates:
                本轮参与训练的客户端更新。

        输出：
            {
                client_id: raw_weight
            }

        注意：
            这里返回的是未归一化权重，也就是每个客户端的样本数。
            归一化会在 Aggregator.aggregate() 里统一调用 normalize_weights() 完成。

        示例：
            client 0 有 100 个样本
            client 1 有 300 个样本

            这里返回：
                client 0 -> 100
                client 1 -> 300

            后续归一化后：
                client 0 -> 0.25
                client 1 -> 0.75
        """
        return build_sample_weights(client_updates)