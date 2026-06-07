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

    round_results: 每一轮训练后的结果摘要。
    train_state: 训练结束后的服务端状态。
    best_acc: 历史最佳测试准确率。
    best_round: 历史最佳准确率对应的轮数。
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

        logging_cfg = _cfg_get(self.cfg, "logging", {})
        log_every = int(_cfg_get(logging_cfg, "log_every", 1))

        # 控制台进度条开关。
        # progress_bar: 是否启用 tqdm。
        # progress_in_non_tty: 是否允许在 nohup / 重定向等非交互终端里显示进度条。
        progress_bar_enabled = _cfg_get_bool(
            logging_cfg,
            "progress_bar",
            True,
        )
        progress_in_non_tty = _cfg_get_bool(
            logging_cfg,
            "progress_in_non_tty",
            False,
        )

        # 控制台只打印短摘要，方便实时观察。
        console_round_summary = _cfg_get_bool(
            logging_cfg,
            "console_round_summary",
            True,
        )

        # train.log 写更详细的结构化日志，但不写进度条。
        file_round_detail = _cfg_get_bool(
            logging_cfg,
            "file_round_detail",
            True,
        )

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

        # 进度条只写到 Python 原始 stderr。
        # 这样即使 train.py 用 tee_output_to_file() 捕获 stdout / stderr，
        # tqdm 进度条也尽量不会进入 train.log。
        #
        # 注意：
        # utils/logging.py 里也建议对 stderr 做 tqdm 过滤，
        # 这里和 TeeStream 过滤是双保险。
        progress_file = getattr(sys, "__stderr__", sys.stderr)
        progress_is_tty = bool(
            getattr(progress_file, "isatty", lambda: False)()
        )
        progress_enabled = bool(progress_bar_enabled) and (
            progress_is_tty or bool(progress_in_non_tty)
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

                    if not progress_bar.disable:
                        progress_bar.set_postfix(
                            round=f"{round_id}/{rounds}",
                            client=int(client.client_id),
                            best=f"{self.train_state.best_acc:.2f}%",
                            refresh=False,
                        )
                        progress_bar.update(len(single_client_updates))

                # 聚合前先清掉控制台进度条，避免 print 的每轮摘要和 tqdm 混在一起。
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

                if not progress_bar.disable:
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

                    # 控制台短摘要：给人实时看。
                    # 由于 train.py 的 tee 会双写 stdout，
                    # 这一行也会进入 train.log，作为每轮简洁摘要。
                    if console_round_summary:
                        self.print_round_summary(round_result)

                    # 文件详细日志：只写 train.log，不污染控制台。
                    if file_round_detail:
                        self.print_file_round_detail(round_result)

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

        这部分同时进入控制台和 train.log。
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
        打印每轮训练短摘要。

        这部分适合控制台实时观察，所以尽量短。
        因为 train.py 使用 tee 输出，所以这一行也会进入 train.log。
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

    def print_file_round_detail(
        self,
        round_result: RoundResult,
    ) -> None:
        """
        写入每轮详细日志。

        这部分只写 train.log，不打印到控制台。
        目的：
        1. 控制台保持干净，只看每轮核心指标。
        2. train.log 保留更完整的信息，方便复盘。
        3. 不写 tqdm 进度条。
        """
        logging_cfg = _cfg_get(self.cfg, "logging", {})
        log_client_metrics = _cfg_get_bool(
            logging_cfg,
            "log_client_metrics",
            True,
        )
        log_agg_weights = _cfg_get_bool(
            logging_cfg,
            "log_agg_weights",
            True,
        )

        avg_train_loss = round_result.aggregation_info.get(
            "avg_train_loss",
            None,
        )
        avg_train_acc = round_result.aggregation_info.get(
            "avg_train_acc",
            None,
        )

        avg_train_loss_text = self._format_metric(
            avg_train_loss,
            fmt=".4f",
        )
        avg_train_acc_text = self._format_metric(
            avg_train_acc,
            fmt=".2f",
            suffix="%",
        )

        self._write_log_only(
            f"[RoundClients] "
            f"round={round_result.round_id} "
            f"selected_clients={round_result.selected_clients}"
        )

        self._write_log_only(
            f"[RoundMetrics] "
            f"round={round_result.round_id} "
            f"train_loss={avg_train_loss_text} "
            f"train_acc={avg_train_acc_text} "
            f"test_loss={round_result.test_loss:.4f} "
            f"test_acc={round_result.test_acc:.2f}% "
            f"best_acc={round_result.best_acc:.2f}%"
        )

        if log_client_metrics:
            self._write_client_metrics_to_log(round_result)

        self._write_aggregation_info_to_log(
            round_result=round_result,
            log_agg_weights=log_agg_weights,
        )

    def _write_client_metrics_to_log(
        self,
        round_result: RoundResult,
    ) -> None:
        """
        写入每个客户端的训练指标。

        collect_client_metrics() 的具体字段由 fl/types.py 决定，
        这里不强行假设字段名，避免和已有结构冲突。
        """
        client_metrics = getattr(round_result, "client_metrics", [])

        for item in client_metrics:
            self._write_log_only(
                f"[ClientMetrics] "
                f"round={round_result.round_id} "
                f"{self._compact_log_value(item)}"
            )

    def _write_aggregation_info_to_log(
        self,
        round_result: RoundResult,
        log_agg_weights: bool,
    ) -> None:
        """
        写入聚合器摘要信息。

        如果聚合器 summary 里包含 weights / client_weights，
        则在 log_agg_weights=true 时额外打印权重。
        """
        for group_name in ("non_expert", "expert"):
            summary = round_result.aggregation_info.get(group_name, None)
            if summary is None:
                continue

            if not isinstance(summary, dict):
                self._write_log_only(
                    f"[AggSummary][{group_name}] "
                    f"round={round_result.round_id} "
                    f"summary={self._compact_log_value(summary)}"
                )
                continue

            method = summary.get(
                "method",
                summary.get(
                    "method_name",
                    summary.get("aggregator", "unknown"),
                ),
            )
            num_clients = summary.get(
                "num_clients",
                summary.get(
                    "effective_clients",
                    summary.get("num_effective_clients", "unknown"),
                ),
            )

            self._write_log_only(
                f"[AggSummary][{group_name}] "
                f"round={round_result.round_id} "
                f"method={method} "
                f"num_clients={num_clients}"
            )

            weights = None
            for weight_key in (
                "weights",
                "client_weights",
                "sample_weights",
                "effective_weights",
            ):
                if weight_key in summary:
                    weights = summary[weight_key]
                    break

            if log_agg_weights and weights is not None:
                self._write_log_only(
                    f"[AggWeights][{group_name}] "
                    f"round={round_result.round_id} "
                    f"weights={self._compact_log_value(weights)}"
                )

            # 额外字段也写入日志，但避免重复打印权重类字段。
            ignored_keys = {
                "weights",
                "client_weights",
                "sample_weights",
                "effective_weights",
            }
            extra_summary = {
                key: value
                for key, value in summary.items()
                if key not in ignored_keys
            }

            if extra_summary:
                self._write_log_only(
                    f"[AggDetail][{group_name}] "
                    f"round={round_result.round_id} "
                    f"summary={self._compact_log_value(extra_summary)}"
                )

    def _write_log_only(self, message: str) -> None:
        """
        只写入 train.log，不打印到控制台。

        原理：
        utils/logging.py 的 TeeStream 会在 sys.stdout 上保存 log_file。
        如果当前确实处于 tee_output_to_file() 环境中，就直接写 log_file。
        如果没有使用 tee，则退化为普通 print，避免信息丢失。
        """
        stdout = sys.stdout
        log_file = getattr(stdout, "log_file", None)
        lock = getattr(stdout, "lock", None)

        if log_file is None:
            print(message)
            return

        if lock is None:
            log_file.write(message + "\n")
            log_file.flush()
            return

        with lock:
            log_file.write(message + "\n")
            log_file.flush()

    @staticmethod
    def _format_metric(
        value: Any,
        *,
        fmt: str,
        suffix: str = "",
    ) -> str:
        """
        格式化日志里的指标。

        value 为 None 时写 nan。
        """
        if value is None:
            return "nan"

        try:
            return f"{float(value):{fmt}}{suffix}"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _compact_log_value(
        value: Any,
        *,
        max_chars: int = 1200,
    ) -> str:
        """
        把日志字段压成一行，避免 train.log 被超长对象刷屏。
        """
        text = repr(value)
        if len(text) <= max_chars:
            return text

        return text[:max_chars] + "...<truncated>"

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
                "请检查模型参数名是否包含 experts.。"
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

    dict 或 ConfigNode: cfg.get(key, default)
    普通对象: getattr(cfg, key, default)
    """
    if hasattr(cfg, "get"):
        return cfg.get(key, default)

    return getattr(cfg, key, default)


def _cfg_get_bool(
    cfg: Any,
    key: str,
    default: bool = False,
) -> bool:
    """
    从配置里读取 bool 值。

    支持 true / false、1 / 0、yes / no、on / off 等常见写法。
    """
    value = _cfg_get(cfg, key, default)

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False

    return bool(value)