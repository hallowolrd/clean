from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from aggregation.factory import build_aggregators
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
        """转成普通 dict，方便后续保存日志。"""
        return {
            "best_acc": float(self.best_acc),
            "best_round": int(self.best_round),
            "round_results": [item.to_dict() for item in self.round_results],
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
        progress_is_tty = bool(getattr(progress_file, "isatty", lambda: False)())
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
        """构建单轮训练结果摘要。"""
        selected_client_ids = [int(client.client_id) for client in selected_clients]

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

        # 保存每个客户端的轻量诊断信息。
        # 注意：这里不保存 model_delta，也不保存 expert_kfac 原始矩阵，
        # 避免 summary.json / train.log 过大。
        full_aggregation_info["client_diagnostics"] = self._build_client_diagnostics(
            client_updates
        )

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
        当前记录：
        1. 本轮整体 train/test 指标
        2. 本轮选择的客户端
        3. non_expert / expert 分别用的聚合方法
        4. non_expert / expert 每个客户端的聚合权重
        5. 每个客户端样本数、本地 train_loss/train_acc、expert_usage
        """
        logging_cfg = _cfg_get(self.cfg, "logging", {})
        log_round_clients = _cfg_get_bool(
            logging_cfg,
            "log_round_clients",
            True,
        )
        log_client_table = _cfg_get_bool(
            logging_cfg,
            "log_client_table",
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
            f"[RoundMetrics] "
            f"round={round_result.round_id} "
            f"train_loss={avg_train_loss_text} "
            f"train_acc={avg_train_acc_text} "
            f"test_loss={round_result.test_loss:.4f} "
            f"test_acc={round_result.test_acc:.2f}% "
            f"best_acc={round_result.best_acc:.2f}%"
        )

        if log_round_clients:
            self._write_log_only(
                f"[Clients] "
                f"round={round_result.round_id} "
                f"ids={self._format_client_ids(round_result.selected_clients)}"
            )

        # 聚合器摘要：方法、客户端数、参数数量、权重。
        self._write_aggregation_info_to_log(
            round_result=round_result,
            log_agg_weights=log_agg_weights,
        )

        # 每个客户端一行诊断信息：样本数、训练指标、聚合权重、expert usage。
        if log_client_table:
            self._write_client_table_to_log(round_result)

    def _build_client_diagnostics(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> Dict[int, Dict[str, Any]]:
        """
        从 ClientUpdate 中提取轻量客户端诊断信息。

        保留：
        1. num_samples
        2. metrics: train_loss / train_acc / num_batches
        3. expert_usage: 每个 expert 的激活次数与比例

        不保留：
        1. model_delta
        2. expert_kfac 原始矩阵

        这样日志和 summary.json 不会被大对象撑爆。
        """
        diagnostics: Dict[int, Dict[str, Any]] = {}

        for update in client_updates:
            extra = dict(update.extra or {})
            diagnostics[int(update.client_id)] = {
                "num_samples": int(update.num_samples),
                "metrics": dict(update.metrics or {}),
                "expert_usage": extra.get("expert_usage", None),
                "expert_kfac_summary": extra.get("expert_kfac_summary", None),
            }

        return diagnostics

    def _write_aggregation_info_to_log(
        self,
        round_result: RoundResult,
        log_agg_weights: bool,
    ) -> None:
        """
        写入 non_expert / expert 聚合摘要。

        输出示例：
        [Agg][non_expert] round=1 method=uniform clients=10 params=121 weights=uniform(each=0.1000)
        [Agg][expert] round=1 method=fisher_kfac_expert clients=10 params=16 weights={0:0.0812,1:0.1033,...}
        """
        logging_cfg = _cfg_get(self.cfg, "logging", {})
        compact_uniform_weights = _cfg_get_bool(
            logging_cfg,
            "compact_uniform_weights",
            True,
        )
        log_history_wolf_kfac_detail = _cfg_get_bool(
            logging_cfg,
            "log_history_wolf_kfac_detail",
            True,
        )

        for group_name in ("non_expert", "expert"):
            agg_info = self._extract_aggregation_info(
                round_result=round_result,
                group_name=group_name,
            )
            if agg_info is None:
                continue

            if log_agg_weights:
                weights_text = self._format_weights(
                    agg_info.get("weights", None),
                    compact_uniform=compact_uniform_weights,
                )
            else:
                weights_text = "hidden"

            method = str(agg_info.get("method", "unknown"))

            self._write_log_only(
                f"[Agg][{group_name}] "
                f"round={round_result.round_id} "
                f"method={method} "
                f"clients={agg_info.get('num_clients', 'unknown')} "
                f"params={agg_info.get('param_count', 'unknown')} "
                f"weights={weights_text}"
            )

            # History-WoLF K-FAC Score 的关键诊断量比较多，单独展开成几行。
            # 这样不用去翻完整 diagnostics dict，也能直接判断超参数是否偏大/偏小。
            if (
                group_name == "expert"
                and method == "history_wolf_kfac_score"
                and log_history_wolf_kfac_detail
            ):
                self._write_history_wolf_kfac_diagnostics_to_log(
                    round_id=round_result.round_id,
                    agg_info=agg_info,
                )

    def _write_history_wolf_kfac_diagnostics_to_log(
        self,
        round_id: int,
        agg_info: Mapping[str, Any],
    ) -> None:
        """
        写入 History-WoLF K-FAC Score 的核心诊断值。

        这些值主要用来判断以下超参数是否设置合理：
        - min_active_count / min_valid_clients / fallback
        - active_count_ref
        - tau_cur / tau_hist
        - c_wolf
        - min_obs_scale
        - q_scale / init_P / seen_ref
        - 最终 expert 内 client 权重是否过于均匀或过于尖锐
        """
        diagnostics = agg_info.get("diagnostics", {})
        if not isinstance(diagnostics, Mapping):
            return

        prefix = f"[AggDiag][history_wolf_kfac_score] round={int(round_id)}"

        # valid / fallback / active_count_ref / route_quality
        self._write_log_only(
            f"{prefix} "
            f"fallback={self._fmt_diag(diagnostics, 'fallback_expert_count', '.0f')} "
            f"fallback_ratio={self._fmt_diag(diagnostics, 'fallback_expert_ratio', '.3f')} "
            f"valid_mean={self._fmt_diag(diagnostics, 'mean_valid_clients', '.2f')} "
            f"valid_min={self._fmt_diag(diagnostics, 'min_valid_clients_observed', '.0f')} "
            f"active_med={self._fmt_diag(diagnostics, 'active_count_median', '.2f')} "
            f"active_mean={self._fmt_diag(diagnostics, 'active_count_mean', '.2f')} "
            f"route_q={self._fmt_diag(diagnostics, 'mean_route_quality', '.3f')} "
            f"route_lt_0.5={self._fmt_diag(diagnostics, 'frac_route_quality_lt_0_5', '.3f')}"
        )

        # tau_cur / tau_hist / c_wolf
        self._write_log_only(
            f"{prefix} "
            f"cur_q={self._fmt_diag(diagnostics, 'mean_current_quality', '.3f')} "
            f"cur_lt_0.2={self._fmt_diag(diagnostics, 'frac_current_quality_lt_0_2', '.3f')} "
            f"hist_q={self._fmt_diag(diagnostics, 'mean_history_quality', '.3f')} "
            f"hist_lt_0.2={self._fmt_diag(diagnostics, 'frac_history_quality_lt_0_2', '.3f')} "
            f"resid_mean={self._fmt_diag(diagnostics, 'residual_dist_mean', '.3f')} "
            f"resid_med={self._fmt_diag(diagnostics, 'residual_dist_median', '.3f')} "
            f"resid_gt_c={self._fmt_diag(diagnostics, 'frac_residual_gt_c_wolf', '.3f')} "
            f"wolf_raw={self._fmt_diag(diagnostics, 'wolf_raw_mean', '.3f')} "
            f"wolf_eff={self._fmt_diag(diagnostics, 'wolf_eff_mean', '.3f')} "
            f"wolf_eff_lt_0.5={self._fmt_diag(diagnostics, 'frac_wolf_eff_lt_0_5', '.3f')}"
        )

        # min_obs_scale / Kalman / P / history_conf / 最终权重
        self._write_log_only(
            f"{prefix} "
            f"obs_scale={self._fmt_diag(diagnostics, 'obs_scale_mean', '.4f')} "
            f"obs_floor={self._fmt_diag(diagnostics, 'frac_obs_scale_at_floor', '.3f')} "
            f"K={self._fmt_diag(diagnostics, 'kalman_gain_mean', '.3f')} "
            f"K_lt_0.1={self._fmt_diag(diagnostics, 'frac_K_lt_0_1', '.3f')} "
            f"K_gt_0.8={self._fmt_diag(diagnostics, 'frac_K_gt_0_8', '.3f')} "
            f"P_new={self._fmt_diag(diagnostics, 'P_new_mean', '.4g')} "
            f"P_shrink={self._fmt_diag(diagnostics, 'P_shrink_ratio_mean', '.3f')} "
            f"seen={self._fmt_diag(diagnostics, 'seen_mean', '.2f')} "
            f"cold={self._fmt_diag(diagnostics, 'frac_cold_start', '.3f')} "
            f"hist_conf={self._fmt_diag(diagnostics, 'history_conf_mean', '.3f')} "
            f"ess={self._fmt_diag(diagnostics, 'mean_weight_ess', '.2f')} "
            f"top1={self._fmt_diag(diagnostics, 'mean_top1_weight', '.3f')} "
            f"entropy={self._fmt_diag(diagnostics, 'mean_weight_normalized_entropy', '.3f')}"
        )

        # 每个 expert 的精简诊断。
        # 只打印一行，避免 train.log 被 per-client 中间量刷屏。
        per_expert_debug_text = self._format_history_wolf_per_expert_debug(
            diagnostics.get("per_expert_debug", None)
        )
        if per_expert_debug_text != "none":
            self._write_log_only(
                f"[AggDiagExpert][history_wolf_kfac_score] "
                f"round={int(round_id)} "
                f"{per_expert_debug_text}"
            )

    def _write_client_table_to_log(
        self,
        round_result: RoundResult,
    ) -> None:
        """
        写入每个客户端的一行诊断信息。

        每行包含：
        1. 客户端样本数
        2. 客户端本地 train_loss / train_acc
        3. non_expert 聚合权重
        4. expert 聚合权重
        5. expert_usage
        """
        client_diagnostics = round_result.aggregation_info.get(
            "client_diagnostics",
            {},
        )
        if not isinstance(client_diagnostics, Mapping):
            return

        non_expert_info = self._extract_aggregation_info(
            round_result=round_result,
            group_name="non_expert",
        )
        expert_info = self._extract_aggregation_info(
            round_result=round_result,
            group_name="expert",
        )

        non_expert_weights = {}
        expert_weights = {}
        if non_expert_info is not None:
            non_expert_weights = non_expert_info.get("weights", {}) or {}
        if expert_info is not None:
            expert_weights = expert_info.get("weights", {}) or {}

        for client_id in round_result.selected_clients:
            client_id = int(client_id)
            item = self._get_client_diagnostic(
                client_diagnostics,
                client_id,
            )

            if item is None:
                self._write_log_only(
                    f"[Client][{client_id}] "
                    f"round={round_result.round_id} "
                    f"missing_diagnostics=true"
                )
                continue

            metrics = item.get("metrics", {}) or {}
            num_samples = item.get("num_samples", "unknown")

            train_loss = self._format_metric(
                metrics.get("train_loss", None),
                fmt=".4f",
            )
            train_acc = self._format_metric(
                metrics.get("train_acc", None),
                fmt=".2f",
                suffix="%",
            )

            non_expert_weight = self._format_weight_value(
                self._get_weight_for_client(non_expert_weights, client_id)
            )
            expert_weight = self._format_weight_value(
                self._get_weight_for_client(expert_weights, client_id)
            )

            expert_usage_text = self._format_expert_usage(
                item.get("expert_usage", None)
            )

            self._write_log_only(
                f"[Client][{client_id}] "
                f"round={round_result.round_id} "
                f"samples={num_samples} "
                f"train_loss={train_loss} "
                f"train_acc={train_acc} "
                f"non_expert_w={non_expert_weight} "
                f"expert_w={expert_weight} "
                f"{expert_usage_text}"
            )

    def _extract_aggregation_info(
        self,
        round_result: RoundResult,
        group_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        提取某个参数组的聚合信息。

        当前 AggregationResult.summary() 常见结构：
        {
            "weights": {...},
            "diagnostics": {
                "method": "uniform",
                "param_group": "expert",
                "num_clients": 10,
                "param_count": 16,
                ...
            }
        }

        这个函数会把外层 weights 和内层 diagnostics 合并成一个扁平 dict，
        方便日志打印。
        """
        summary = round_result.aggregation_info.get(group_name, None)
        if summary is None:
            return None

        if not isinstance(summary, Mapping):
            return {
                "method": "unknown",
                "param_group": group_name,
                "num_clients": "unknown",
                "param_count": "unknown",
                "weights": None,
                "raw_summary": summary,
            }

        diagnostics = summary.get("diagnostics", {})
        if not isinstance(diagnostics, Mapping):
            diagnostics = {}

        method = diagnostics.get(
            "method",
            summary.get(
                "method",
                summary.get("method_name", summary.get("aggregator", "unknown")),
            ),
        )
        param_group = diagnostics.get(
            "param_group",
            summary.get("param_group", group_name),
        )
        num_clients = diagnostics.get(
            "num_clients",
            summary.get(
                "num_clients",
                summary.get(
                    "effective_clients",
                    summary.get("num_effective_clients", "unknown"),
                ),
            ),
        )
        param_count = diagnostics.get(
            "param_count",
            summary.get("param_count", "unknown"),
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

        if weights is None:
            for weight_key in (
                "weights",
                "client_weights",
                "sample_weights",
                "effective_weights",
            ):
                if weight_key in diagnostics:
                    weights = diagnostics[weight_key]
                    break

        return {
            "method": method,
            "param_group": param_group,
            "num_clients": num_clients,
            "param_count": param_count,
            "weights": weights,
            "diagnostics": dict(diagnostics),
        }

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
    def _format_weight_value(value: Any) -> str:
        """
        格式化单个客户端权重。
        例如：0.10000000000000002 -> 0.1000
        """
        if value is None:
            return "nan"

        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)

    def _format_weights(
        self,
        weights: Any,
        *,
        compact_uniform: bool,
    ) -> str:
        """
        格式化聚合权重。

        uniform 权重默认压缩成：
            uniform(each=0.1000)

        非 uniform 权重打印成：
            {0:0.1234,1:0.0987,...}
        """
        if weights is None:
            return "none"

        if not isinstance(weights, Mapping):
            return self._compact_log_value(weights)

        if len(weights) == 0:
            return "{}"

        numeric_items = []
        for key, value in weights.items():
            try:
                client_id = int(key)
                weight_value = float(value)
            except (TypeError, ValueError):
                return self._compact_log_value(weights)
            numeric_items.append((client_id, weight_value))

        numeric_items = sorted(numeric_items, key=lambda item: item[0])

        if compact_uniform and self._is_uniform_weight_items(numeric_items):
            return f"uniform(each={numeric_items[0][1]:.4f})"

        body = ",".join(
            f"{client_id}:{weight_value:.4f}"
            for client_id, weight_value in numeric_items
        )
        return "{" + body + "}"

    @staticmethod
    def _is_uniform_weight_items(
        items: Sequence[tuple[int, float]],
        *,
        atol: float = 1.0e-10,
    ) -> bool:
        """
        判断权重是否近似均匀。
        用于把一长串 0.10000000000000002 压缩成 uniform(each=0.1000)。
        """
        if len(items) == 0:
            return False

        first_value = float(items[0][1])
        for _, value in items:
            if abs(float(value) - first_value) > atol:
                return False
        return True

    @staticmethod
    def _format_client_ids(client_ids: Sequence[int]) -> str:
        """
        格式化客户端 id 列表。
        输出：[0,4,9,6]
        """
        body = ",".join(str(int(client_id)) for client_id in client_ids)
        return "[" + body + "]"

    @staticmethod
    def _get_client_diagnostic(
        client_diagnostics: Mapping[Any, Any],
        client_id: int,
    ) -> Optional[Mapping[str, Any]]:
        """兼容 int key / str key 两种客户端诊断字典。"""
        if client_id in client_diagnostics:
            item = client_diagnostics[client_id]
            if isinstance(item, Mapping):
                return item

        str_client_id = str(client_id)
        if str_client_id in client_diagnostics:
            item = client_diagnostics[str_client_id]
            if isinstance(item, Mapping):
                return item

        return None

    @staticmethod
    def _get_weight_for_client(
        weights: Any,
        client_id: int,
    ) -> Optional[Any]:
        """
        从权重字典中读取某个客户端的权重。
        兼容：weights[0] / weights["0"]
        """
        if not isinstance(weights, Mapping):
            return None

        if client_id in weights:
            return weights[client_id]

        str_client_id = str(client_id)
        if str_client_id in weights:
            return weights[str_client_id]

        return None

    def _format_expert_usage(
        self,
        expert_usage: Any,
    ) -> str:
        """
        格式化客户端 expert usage。

        输出示例：
        expert_active=4/4 expert_total=9600 expert_counts={0:2400,1:2381,2:2410,3:2409} expert_frac={0:0.250,1:0.248,2:0.251,3:0.251}
        """
        if expert_usage is None:
            return "expert_usage=none"

        if not isinstance(expert_usage, Mapping):
            return f"expert_usage={self._compact_log_value(expert_usage)}"

        supported = bool(expert_usage.get("supported", True))
        if not supported:
            reason = expert_usage.get("reason", "unknown")
            return (
                "expert_usage=unsupported"
                f"(reason={self._compact_log_value(reason, max_chars=160)})"
            )

        num_experts = expert_usage.get(
            "num_experts",
            _cfg_get(self.cfg, "num_experts", "unknown"),
        )
        active_experts = expert_usage.get("active_experts", "unknown")
        total_activations = expert_usage.get("total_activations", "unknown")
        expert_counts = expert_usage.get("expert_counts", None)
        expert_fraction = expert_usage.get("expert_fraction", None)
        dead_experts = expert_usage.get("dead_experts", [])

        counts_text = self._format_int_mapping(expert_counts)
        fraction_text = self._format_float_mapping(
            expert_fraction,
            precision=3,
        )

        return (
            f"expert_active={active_experts}/{num_experts} "
            f"expert_total={total_activations} "
            f"expert_counts={counts_text} "
            f"expert_frac={fraction_text} "
            f"dead={self._format_client_ids(dead_experts)}"
        )

    @staticmethod
    def _format_int_mapping(value: Any) -> str:
        """
        格式化 int 映射。
        输出：{0:120,1:130}
        """
        if not isinstance(value, Mapping):
            return "none"

        items = []
        for key, item_value in value.items():
            try:
                items.append((int(key), int(item_value)))
            except (TypeError, ValueError):
                return repr(value)

        items = sorted(items, key=lambda item: item[0])
        body = ",".join(f"{key}:{item_value}" for key, item_value in items)
        return "{" + body + "}"

    @staticmethod
    def _format_float_mapping(
        value: Any,
        *,
        precision: int,
    ) -> str:
        """
        格式化 float 映射。
        输出：{0:0.250,1:0.248}
        """
        if not isinstance(value, Mapping):
            return "none"

        items = []
        for key, item_value in value.items():
            try:
                items.append((int(key), float(item_value)))
            except (TypeError, ValueError):
                return repr(value)

        items = sorted(items, key=lambda item: item[0])
        body = ",".join(
            f"{key}:{item_value:.{precision}f}"
            for key, item_value in items
        )
        return "{" + body + "}"

    @staticmethod
    def _fmt_diag(
        diagnostics: Mapping[str, Any],
        key: str,
        fmt: str,
        default: Any = "nan",
    ) -> str:
        """
        格式化 diagnostics 里的单个数值。

        这个函数只用于日志，不参与算法逻辑。
        """
        value = diagnostics.get(key, default)
        try:
            return f"{float(value):{fmt}}"
        except (TypeError, ValueError):
            return str(value)

    def _format_history_wolf_per_expert_debug(self, value: Any) -> str:
        """
        压缩打印每个 expert 的关键诊断。

        输出示例：
        experts={0:(valid=10,fb=0,rq=0.82,K=0.31,ess=6.7,top1=0.22),...}
        """
        if not isinstance(value, Mapping) or len(value) == 0:
            return "none"

        items = []
        for expert_id, item in value.items():
            if not isinstance(item, Mapping):
                continue
            try:
                expert_idx = int(expert_id)
            except (TypeError, ValueError):
                continue

            valid = item.get("valid_clients", "?")
            fallback = 1 if bool(item.get("fallback", False)) else 0
            route_q = self._format_mapping_float_value(
                item,
                "route_quality_mean",
                precision=3,
            )
            kalman_gain = self._format_mapping_float_value(
                item,
                "kalman_gain_mean",
                precision=3,
            )
            wolf_eff = self._format_mapping_float_value(
                item,
                "wolf_eff_mean",
                precision=3,
            )
            ess = self._format_mapping_float_value(
                item,
                "weight_ess",
                precision=2,
            )
            top1 = self._format_mapping_float_value(
                item,
                "top1_weight",
                precision=3,
            )

            items.append(
                (
                    expert_idx,
                    f"{expert_idx}:(valid={valid},fb={fallback},"
                    f"rq={route_q},wolf={wolf_eff},K={kalman_gain},"
                    f"ess={ess},top1={top1})",
                )
            )

        if len(items) == 0:
            return "none"

        items = sorted(items, key=lambda pair: pair[0])
        return "experts={" + ",".join(text for _, text in items) + "}"

    @staticmethod
    def _format_mapping_float_value(
        value: Mapping[str, Any],
        key: str,
        *,
        precision: int,
    ) -> str:
        """从 mapping 里取一个 float，并按指定精度格式化。"""
        raw_value = value.get(key, None)
        if raw_value is None:
            return "nan"
        try:
            return f"{float(raw_value):.{precision}f}"
        except (TypeError, ValueError):
            return str(raw_value)

    @staticmethod
    def _compact_log_value(
        value: Any,
        *,
        max_chars: int = 1200,
    ) -> str:
        """把日志字段压成一行，避免 train.log 被超长对象刷屏。"""
        text = repr(value)
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."

    def _validate_server_state(self) -> None:
        """检查服务端初始化状态是否合法。"""
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
        """每轮结束后清理显存和 Python 垃圾对象。"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_server(
    cfg: Any,
    client_loaders: Sequence[DataLoader],
    test_loader: DataLoader,
    device: torch.device | str,
) -> FLServer:
    """构建 FLServer。train.py 后面可以直接调用这个函数。"""
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
