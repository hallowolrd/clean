from __future__ import annotations

import gc
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from aggregation.factory import AggregatorBundle, build_aggregators
from fl.client import (
    FLClient,
    build_clients,
    select_clients,
    train_selected_clients,
)
from fl.types import (
    ClientUpdate,
    RoundResult,
    TrainState,
    average_client_metric,
    collect_client_metrics,
)
from models.build import build_model, summarize_model
from models.param_groups import (
    ParamGroups,
    build_param_groups,
    summarize_param_groups,
)
from utils.eval import EvalResult, evaluate
from utils.state_dict_ops import (
    check_finite_state_dict,
    state_dict_to,
)


@dataclass
class ServerTrainResult:
    """
    服务端完整训练结果。

    round_results:
        每一轮训练后的结果摘要。

    train_state:
        训练结束后的服务端状态。

    best_acc:
        历史最佳测试准确率。

    best_round:
        历史最佳准确率对应的轮数。
    """

    round_results: List[RoundResult]
    train_state: TrainState
    best_acc: float
    best_round: int

    def to_dict(self) -> Dict[str, Any]:
        """
        转成普通 dict，方便后续保存日志。
        """
        return {
            "best_acc": float(self.best_acc),
            "best_round": int(self.best_round),
            "round_results": [
                item.to_dict()
                for item in self.round_results
            ],
            "train_state": self.train_state.to_dict(),
        }


class FLServer:
    """
    联邦学习服务端。

    职责：
        1. 持有全局模型
        2. 选择每轮参与训练的客户端
        3. 收集客户端更新
        4. 分别聚合 non_expert 参数和 expert 参数
        5. 在服务器测试集上评估全局模型

    不负责：
        1. 数据集加载
        2. 数据划分
        3. 具体客户端本地训练细节
        4. 具体聚合权重算法细节
        5. checkpoint 保存
    """

    def __init__(
        self,
        cfg: Any,
        client_loaders: Sequence[DataLoader],
        test_loader: DataLoader,
        device: torch.device | str,
        global_model: Optional[nn.Module] = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)

        # 全局模型默认常驻 CPU。
        # 客户端训练时会 deepcopy 后移动到 GPU。
        # 服务端评估时临时移动到 GPU，评估后再移回 CPU。
        if global_model is None:
            self.global_model = build_model(cfg)
        else:
            self.global_model = global_model

        self.global_model.to("cpu")

        self.clients = build_clients(
            cfg=cfg,
            client_loaders=client_loaders,
            device=self.device,
        )

        self.test_loader = test_loader

        self.aggregators = build_aggregators(cfg)

        self.param_groups = build_param_groups(
            model=self.global_model,
            expected_num_experts=int(_cfg_get(cfg, "num_experts", 0)),
            strict=True,
        )

        self.train_state = TrainState(
            round_id=0,
            best_acc=0.0,
            best_round=0,
            extra={},
        )

        self.round_results: List[RoundResult] = []

        self._validate_server_state()

    def train(self) -> ServerTrainResult:
        """
        执行完整 FL 训练流程。

        每一轮流程：
            1. 选择客户端
            2. 客户端本地训练
            3. 聚合 non_expert 参数
            4. 聚合 expert 参数
            5. 更新全局模型
            6. 在服务器测试集评估
            7. 记录 RoundResult
        """
        rounds = int(_cfg_get(self.cfg, "rounds", 1))
        frac = float(_cfg_get(self.cfg, "frac", 1.0))
        seed = int(_cfg_get(self.cfg, "seed", 42))
        log_every = int(_cfg_get(_cfg_get(self.cfg, "logging", {}), "log_every", 1))

        if rounds <= 0:
            raise ValueError(f"rounds 必须大于 0，当前值：{rounds}")

        self.print_startup_summary()

        # 预估整个实验的客户端训练步数。
        # 进度条单位：完成一个客户端本地训练。
        total_client_steps = sum(
            len(
                select_clients(
                    clients=self.clients,
                    frac=frac,
                    round_id=progress_round_id,
                    seed=seed,
                )
            )
            for progress_round_id in range(1, rounds + 1)
        )

        # 进度条只写到 Python 原始 stderr，尽量绕开 train.py 里的 tee 日志捕获。
        # 如果当前不是交互式终端，比如 nohup / 重定向 / 日志文件，则自动关闭进度条。
        progress_file = getattr(sys, "__stderr__", sys.stderr)
        progress_enabled = bool(
            getattr(progress_file, "isatty", lambda: False)()
        )

        progress_bar = tqdm(
            total=total_client_steps,
            desc="Training",
            dynamic_ncols=True,
            leave=True,
            file=progress_file,
            disable=not progress_enabled,
            mininterval=0.5,
        )

        try:
            for round_id in range(1, rounds + 1):
                selected_clients = select_clients(
                    clients=self.clients,
                    frac=frac,
                    round_id=round_id,
                    seed=seed,
                )

                client_updates: List[ClientUpdate] = []

                # 逐客户端训练，保证每完成一个客户端本地训练就更新一次总进度条。
                for client in selected_clients:
                    single_client_updates = train_selected_clients(
                        clients=[client],
                        global_model=self.global_model,
                        round_id=round_id,
                    )

                    client_updates.extend(single_client_updates)

                    progress_bar.set_postfix(
                        round=f"{round_id}/{rounds}",
                        client=int(client.client_id),
                        best=f"{self.train_state.best_acc:.2f}%",
                        refresh=False,
                    )
                    progress_bar.update(len(single_client_updates))

                if not progress_bar.disable:
                    progress_bar.clear()

                aggregation_info = self.aggregate_client_updates(
                    client_updates=client_updates,
                )

                if not progress_bar.disable:
                    progress_bar.refresh()

                eval_result = self.evaluate_global_model()

                if eval_result.acc > self.train_state.best_acc:
                    self.train_state.best_acc = float(eval_result.acc)
                    self.train_state.best_round = int(round_id)

                self.train_state.round_id = int(round_id)

                round_result = self.build_round_result(
                    round_id=round_id,
                    selected_clients=selected_clients,
                    client_updates=client_updates,
                    eval_result=eval_result,
                    aggregation_info=aggregation_info,
                )

                self.round_results.append(round_result)

                avg_train_loss = round_result.aggregation_info.get(
                    "avg_train_loss",
                    None,
                )
                if avg_train_loss is None:
                    avg_train_loss_text = "nan"
                else:
                    avg_train_loss_text = f"{avg_train_loss:.4f}"

                progress_bar.set_postfix(
                    round=f"{round_id}/{rounds}",
                    client="done",
                    acc=f"{eval_result.acc:.2f}%",
                    best=f"{self.train_state.best_acc:.2f}%",
                    loss=avg_train_loss_text,
                    refresh=False,
                )
                progress_bar.refresh()

                if log_every > 0 and round_id % log_every == 0:
                    if not progress_bar.disable:
                        progress_bar.clear()

                    self.print_round_summary(round_result)

                    if not progress_bar.disable:
                        progress_bar.refresh()

                self._cleanup_after_round()

        finally:
            progress_bar.close()

        return ServerTrainResult(
            round_results=list(self.round_results),
            train_state=self.train_state,
            best_acc=float(self.train_state.best_acc),
            best_round=int(self.train_state.best_round),
        )

    def aggregate_client_updates(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> Dict[str, Any]:
        """
        聚合客户端更新。

        聚合顺序：
            1. non_expert 参数
            2. expert 参数

        这样可以让二者使用不同聚合器：
            non_expert: sample_weighted / uniform
            expert: uniform / sample_weighted / 后续 Fisher / Bayes
        """
        if len(client_updates) == 0:
            raise ValueError("client_updates 不能为空。")

        global_state_cpu = state_dict_to(
            self.global_model.state_dict(),
            device="cpu",
        )

        non_expert_result = self.aggregators.non_expert.aggregate(
            global_state=global_state_cpu,
            client_updates=client_updates,
            param_names=self.param_groups.non_expert,
            base_state=None,
            strict=True,
        )

        expert_result = self.aggregators.expert.aggregate(
            global_state=global_state_cpu,
            client_updates=client_updates,
            param_names=self.param_groups.expert,
            base_state=non_expert_result.new_state_dict,
            strict=True,
        )

        new_state_dict = expert_result.new_state_dict

        check_finite_state_dict(new_state_dict)

        self.global_model.load_state_dict(
            new_state_dict,
            strict=True,
        )
        self.global_model.to("cpu")

        return {
            "non_expert": non_expert_result.summary(),
            "expert": expert_result.summary(),
        }

    def evaluate_global_model(self) -> EvalResult:
        """
        在服务器测试集上评估全局模型。

        注意：
            测试集只在服务器使用。
            不参与客户端训练。
            不参与参数聚合。
        """
        self.global_model.to(self.device)

        result = evaluate(
            model=self.global_model,
            data_loader=self.test_loader,
            device=self.device,
        )

        self.global_model.to("cpu")

        return result

    def build_round_result(
        self,
        round_id: int,
        selected_clients: Sequence[FLClient],
        client_updates: Sequence[ClientUpdate],
        eval_result: EvalResult,
        aggregation_info: Dict[str, Any],
    ) -> RoundResult:
        """
        构建单轮训练结果摘要。
        """
        selected_client_ids = [
            int(client.client_id)
            for client in selected_clients
        ]

        avg_train_loss = average_client_metric(
            client_updates=list(client_updates),
            metric_name="train_loss",
            weighted=True,
            default=None,
        )
        avg_train_acc = average_client_metric(
            client_updates=list(client_updates),
            metric_name="train_acc",
            weighted=True,
            default=None,
        )

        full_aggregation_info = dict(aggregation_info)
        full_aggregation_info["avg_train_loss"] = avg_train_loss
        full_aggregation_info["avg_train_acc"] = avg_train_acc

        return RoundResult(
            round_id=int(round_id),
            selected_clients=selected_client_ids,
            test_loss=float(eval_result.loss),
            test_acc=float(eval_result.acc),
            best_acc=float(self.train_state.best_acc),
            client_metrics=collect_client_metrics(list(client_updates)),
            aggregation_info=full_aggregation_info,
        )

    def print_startup_summary(self) -> None:
        """
        打印训练开始前的摘要信息。
        """
        model_summary = summarize_model(self.global_model)
        param_summary = summarize_param_groups(
            state_dict=self.global_model.state_dict(),
            groups=self.param_groups,
        )

        print()
        print("=" * 80)
        print("[Server] FL training start")
        print(f"[Server] device: {self.device}")
        print(f"[Server] num_clients: {len(self.clients)}")
        print(f"[Server] rounds: {int(_cfg_get(self.cfg, 'rounds', 1))}")
        print(f"[Server] frac: {float(_cfg_get(self.cfg, 'frac', 1.0))}")
        print(f"[Server] model: {_cfg_get(self.cfg, 'model', 'unknown')}")
        print(
            "[Server] params: "
            f"total={model_summary['total_params']:,}, "
            f"trainable={model_summary['trainable_params']:,}"
        )
        print(
            "[Server] aggregators: "
            f"non_expert={self.aggregators.non_expert.method_name}, "
            f"expert={self.aggregators.expert.method_name}"
        )
        print(f"[Server] param_groups: {self.param_groups.summary()}")
        print(f"[Server] param_numel: {param_summary['floating_numel']}")
        print("=" * 80)
        print()

    def print_round_summary(
        self,
        round_result: RoundResult,
    ) -> None:
        """
        打印每轮训练摘要。
        """
        avg_train_loss = round_result.aggregation_info.get(
            "avg_train_loss",
            None,
        )
        avg_train_acc = round_result.aggregation_info.get(
            "avg_train_acc",
            None,
        )

        if avg_train_loss is None:
            avg_train_loss_text = "nan"
        else:
            avg_train_loss_text = f"{avg_train_loss:.4f}"

        if avg_train_acc is None:
            avg_train_acc_text = "nan"
        else:
            avg_train_acc_text = f"{avg_train_acc:.2f}%"

        print(
            f"[Round {round_result.round_id:03d}] "
            f"train_loss={avg_train_loss_text} | "
            f"train_acc={avg_train_acc_text} | "
            f"test_loss={round_result.test_loss:.4f} | "
            f"test_acc={round_result.test_acc:.2f}% | "
            f"best_acc={round_result.best_acc:.2f}%"
        )

    def _validate_server_state(self) -> None:
        """
        检查服务端初始化状态是否合法。
        """
        if len(self.clients) == 0:
            raise ValueError("服务端没有任何客户端。")

        if self.test_loader is None:
            raise ValueError("test_loader 不能为空。")

        if len(self.param_groups.expert) == 0:
            raise ValueError(
                "没有找到 expert 参数。"
                "请检查模型参数名是否包含 experts.<id>。"
            )

        if len(self.param_groups.non_expert) == 0:
            raise ValueError("没有找到 non_expert 参数。")

    @staticmethod
    def _cleanup_after_round() -> None:
        """
        每轮结束后清理显存和 Python 垃圾对象。
        """
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_server(
    cfg: Any,
    client_loaders: Sequence[DataLoader],
    test_loader: DataLoader,
    device: torch.device | str,
) -> FLServer:
    """
    构建 FLServer。

    train.py 后面可以直接调用这个函数。
    """
    return FLServer(
        cfg=cfg,
        client_loaders=client_loaders,
        test_loader=test_loader,
        device=device,
    )


def resolve_device(cfg: Any) -> torch.device:
    """
    根据 cfg.device 解析训练设备。

    支持：
        auto
        cpu
        cuda
        mps
    """
    device_name = str(_cfg_get(cfg, "device", "auto")).lower().strip()

    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("配置 device=cuda，但当前环境 CUDA 不可用。")

        return torch.device("cuda")

    if device_name == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("配置 device=mps，但当前环境 MPS 不可用。")

        return torch.device("mps")

    if device_name == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"不支持的 device：{device_name}。"
        "当前支持：auto, cpu, cuda, mps"
    )


def _cfg_get(
    cfg: Any,
    key: str,
    default: Any = None,
) -> Any:
    """
    兼容 dict / ConfigNode / 普通对象的读取。

    dict 或 ConfigNode:
        cfg.get(key, default)

    普通对象:
        getattr(cfg, key, default)
    """
    if hasattr(cfg, "get"):
        return cfg.get(key, default)

    return getattr(cfg, key, default)