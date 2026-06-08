from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
        client_evidence_loaders: Optional[Sequence[DataLoader]] = None,
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

        # client_evidence_loaders 用于 expert_fisher / fisher_only / fisher_history_wolf：
        # 客户端本地训练完成后，会额外使用 evidence_loader 做一轮
        # 无数据增强的 forward + backward 来统计 expert K-FAC。
        # server 只负责把 evidence_loader 透传给 client，不关心 Fisher 细节。
        self.clients = build_clients(
            cfg=cfg,
            client_loaders=client_loaders,
            client_evidence_loaders=client_evidence_loaders,
            device=self.device,
        )

        self.test_loader = test_loader
        self.aggregators: AggregatorBundle = build_aggregators(cfg)

        self.param_groups: ParamGroups = build_param_groups(
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

        for round_id in range(1, rounds + 1):
            selected_clients = select_clients(
                clients=self.clients,
                frac=frac,
                round_id=round_id,
                seed=seed,
            )

            client_updates = train_selected_clients(
                clients=selected_clients,
                global_model=self.global_model,
                round_id=round_id,
            )

            aggregation_info = self.aggregate_client_updates(
                client_updates=client_updates,
            )

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

            if log_every > 0 and round_id % log_every == 0:
                self.print_round_summary(round_result)

            # expert 聚合诊断打印由对应配置块控制：
            # - fisher_only 使用 expert_fisher.diagnostics_print
            # - fisher_history_wolf 使用 fisher_history_wolf.diagnostics_print
            #
            # server 这里只读取聚合器已经生成的 diagnostics 并打印摘要，
            # 不在 server 里计算 Fisher / WoLF 细节，保持极致解耦。
            self.print_expert_aggregation_diagnostics(round_result)

            self._cleanup_after_round()

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
            expert: uniform / sample_weighted / fisher_only / fisher_history_wolf
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

        expert_fisher_cfg = _cfg_get(self.cfg, "expert_fisher", {})
        wolf_cfg = _cfg_get(self.cfg, "fisher_history_wolf", {})

        expert_fisher_enabled = bool(_cfg_get(expert_fisher_cfg, "enabled", False))
        fisher_diagnostics_print = bool(
            _cfg_get(expert_fisher_cfg, "diagnostics_print", False)
        )
        wolf_diagnostics_print = bool(
            _cfg_get(wolf_cfg, "diagnostics_print", False)
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
        print(f"[Server] expert_fisher.enabled: {expert_fisher_enabled}")
        print(f"[Server] expert_fisher.diagnostics_print: {fisher_diagnostics_print}")
        print(f"[Server] fisher_history_wolf.diagnostics_print: {wolf_diagnostics_print}")
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

    def print_expert_aggregation_diagnostics(
        self,
        round_result: RoundResult,
    ) -> None:
        """
        打印 expert 聚合器的紧凑诊断日志。

        当前支持：
            fisher_only:
                使用 expert_fisher.diagnostics_print 控制，
                打印前缀默认 [FisherDiag]。

            fisher_history_wolf:
                使用 fisher_history_wolf.diagnostics_print 控制，
                打印前缀默认 [FisherWolfDiag]。

        注意：
            这里只打印 diagnostics 中已经存在的摘要字段。
            Fisher / WoLF 具体计算仍然放在对应 aggregation/*.py 里。
        """
        expert_summary = round_result.aggregation_info.get("expert", {})
        if not isinstance(expert_summary, dict):
            return

        diagnostics = expert_summary.get("diagnostics", {})
        if not isinstance(diagnostics, dict):
            return

        method = diagnostics.get("method", None)

        if method == "fisher_only":
            self._print_fisher_only_diagnostics(
                round_result=round_result,
                diagnostics=diagnostics,
            )
            return

        if method == "fisher_history_wolf":
            self._print_fisher_history_wolf_diagnostics(
                round_result=round_result,
                diagnostics=diagnostics,
            )
            return

    def print_fisher_diagnostics(
        self,
        round_result: RoundResult,
    ) -> None:
        """
        兼容旧调用名。

        新代码统一使用 print_expert_aggregation_diagnostics()。
        """
        self.print_expert_aggregation_diagnostics(round_result)

    def _print_fisher_only_diagnostics(
        self,
        round_result: RoundResult,
        diagnostics: Dict[str, Any],
    ) -> None:
        """
        打印 fisher_only 的紧凑诊断日志。
        """
        expert_fisher_cfg = _cfg_get(self.cfg, "expert_fisher", {})

        diagnostics_print = bool(
            _cfg_get(expert_fisher_cfg, "diagnostics_print", False)
        )
        if not diagnostics_print:
            return

        print_every = int(
            _cfg_get(expert_fisher_cfg, "diagnostics_print_every", 1)
        )
        print_every = max(print_every, 1)

        if int(round_result.round_id) % print_every != 0:
            return

        if not bool(diagnostics.get("fisher_diag_enabled", False)):
            return

        prefix = str(
            _cfg_get(expert_fisher_cfg, "diagnostics_prefix", "[FisherDiag]")
        )

        num_experts = _safe_int(diagnostics.get("num_experts", 0))
        num_fallback_experts = _safe_int(
            diagnostics.get("num_fallback_experts", 0)
        )

        print(
            f"{prefix}[Round {round_result.round_id:03d}] "
            f"experts={num_experts} | "
            f"fallback={num_fallback_experts}/{num_experts} | "
            f"fallback_ratio={_fmt_float(diagnostics.get('fallback_ratio', 0.0), 3)} | "
            f"valid={_fmt_float(diagnostics.get('mean_valid_clients', 0.0), 2)} | "
            f"eff_clients={_fmt_float(diagnostics.get('mean_effective_clients', 0.0), 2)} | "
            f"entropy_norm={_fmt_float(diagnostics.get('mean_weight_entropy_norm', 0.0), 3)} | "
            f"w_max={_fmt_float(diagnostics.get('mean_weight_max', 0.0), 3)} | "
            f"score_cv={_fmt_float(diagnostics.get('mean_score_cv', 0.0), 3)} | "
            f"active_cv={_fmt_float(diagnostics.get('mean_active_count_cv', 0.0), 3)} | "
            f"fisher_cv={_fmt_float(diagnostics.get('mean_fisher_strength_cv', 0.0), 3)} | "
            f"corr_w_active={_fmt_float(diagnostics.get('mean_weight_active_corr', 0.0), 3)} | "
            f"corr_w_fisher={_fmt_float(diagnostics.get('mean_weight_fisher_corr', 0.0), 3)}"
        )

        print_experts = bool(
            _cfg_get(expert_fisher_cfg, "diagnostics_print_experts", False)
        )
        if not print_experts:
            return

        expert_diagnostics = diagnostics.get("expert_diagnostics", {})
        if not isinstance(expert_diagnostics, dict):
            return

        for expert_id, expert_diag in _sorted_expert_diagnostics(
            expert_diagnostics
        ):
            if not isinstance(expert_diag, dict):
                continue

            fallback = bool(expert_diag.get("fallback", False))
            top_client = expert_diag.get("top_client", None)
            top_client_text = "none" if top_client is None else str(top_client)

            print(
                f"{prefix}[Round {round_result.round_id:03d}][Expert {expert_id}] "
                f"valid={_safe_int(expert_diag.get('valid_clients', 0))} | "
                f"fallback={str(fallback).lower()} | "
                f"eff={_fmt_float(expert_diag.get('effective_clients', 0.0), 2)} | "
                f"entropy_norm={_fmt_float(expert_diag.get('weight_entropy_norm', 0.0), 3)} | "
                f"w_max={_fmt_float(expert_diag.get('weight_max', 0.0), 3)} | "
                f"top_client={top_client_text} | "
                f"top1_gap={_fmt_float(expert_diag.get('top1_gap', 0.0), 3)} | "
                f"score_cv={_fmt_float(expert_diag.get('score_cv', 0.0), 3)} | "
                f"active_cv={_fmt_float(expert_diag.get('active_count_cv', 0.0), 3)} | "
                f"fisher_cv={_fmt_float(expert_diag.get('fisher_strength_cv', 0.0), 3)} | "
                f"corr_w_active={_fmt_float(expert_diag.get('weight_active_corr', 0.0), 3)} | "
                f"corr_w_fisher={_fmt_float(expert_diag.get('weight_fisher_corr', 0.0), 3)} | "
                f"zero_score={_safe_int(expert_diag.get('zero_score_clients', 0))} | "
                f"zero_active={_safe_int(expert_diag.get('zero_active_clients', 0))}"
            )

    def _print_fisher_history_wolf_diagnostics(
        self,
        round_result: RoundResult,
        diagnostics: Dict[str, Any],
    ) -> None:
        """
        打印 fisher_history_wolf 的紧凑诊断日志。

        重点看：
            rho:
                WoLF 是否在压异常 Fisher observation。

            kalman_gain:
                历史状态更新速度是否过快 / 过慢。

            abs_resid / abs_resid_p90:
                当前 Fisher observation 和历史预测的偏离幅度。
                如果 abs_resid 很大但 rho 仍高，说明 robust_c 可能太宽松。

            mu_update:
                历史状态这一轮实际改变量。
                如果长期很小，说明历史可能冻结；如果长期很大，说明历史可能乱跳。

            cold_start:
                age==1 的 client-expert 比例。
                中后期如果仍然很高，说明很多 client-expert 没有连续有效 evidence。

            support:
                active_count 低支撑降权是否过强。

            corr_w_active / corr_w_fisher:
                判断权重是否仍主要来自 Fisher，而不是重新被 active_count 支配。
        """
        wolf_cfg = _cfg_get(self.cfg, "fisher_history_wolf", {})

        diagnostics_print = bool(
            _cfg_get(wolf_cfg, "diagnostics_print", False)
        )
        if not diagnostics_print:
            return

        print_every = int(
            _cfg_get(wolf_cfg, "diagnostics_print_every", 1)
        )
        print_every = max(print_every, 1)

        if int(round_result.round_id) % print_every != 0:
            return

        if not bool(diagnostics.get("fisher_wolf_diag_enabled", False)):
            return

        prefix = str(
            _cfg_get(wolf_cfg, "diagnostics_prefix", "[FisherWolfDiag]")
        )

        num_experts = _safe_int(diagnostics.get("num_experts", 0))
        num_fallback_experts = _safe_int(
            diagnostics.get("num_fallback_experts", 0)
        )

        print(
            f"{prefix}[Round {round_result.round_id:03d}] "
            f"experts={num_experts} | "
            f"fallback={num_fallback_experts}/{num_experts} | "
            f"fallback_ratio={_fmt_float(diagnostics.get('fallback_ratio', 0.0), 3)} | "
            f"valid={_fmt_float(diagnostics.get('mean_valid_clients', 0.0), 2)} | "
            f"eff_clients={_fmt_float(diagnostics.get('mean_effective_clients', 0.0), 2)} | "
            f"entropy_norm={_fmt_float(diagnostics.get('mean_weight_entropy_norm', 0.0), 3)} | "
            f"w_max={_fmt_float(diagnostics.get('mean_weight_max', 0.0), 3)} | "
            f"rho={_fmt_float(diagnostics.get('mean_rho', 0.0), 3)} | "
            f"rho_p10={_fmt_float(diagnostics.get('mean_rho_p10', 0.0), 3)} | "
            f"K={_fmt_float(diagnostics.get('mean_kalman_gain', 0.0), 3)} | "
            f"abs_resid={_fmt_float(diagnostics.get('mean_abs_residual', 0.0), 3)} | "
            f"abs_resid_p90={_fmt_float(diagnostics.get('mean_abs_residual_p90', 0.0), 3)} | "
            f"mu_update={_fmt_float(diagnostics.get('mean_mu_update_abs', 0.0), 3)} | "
            f"cold_start={_fmt_float(diagnostics.get('mean_cold_start_frac', 0.0), 3)} | "
            f"support={_fmt_float(diagnostics.get('mean_support', 0.0), 3)} | "
            f"fisher_cv={_fmt_float(diagnostics.get('mean_fisher_strength_cv', 0.0), 3)} | "
            f"active_cv={_fmt_float(diagnostics.get('mean_active_count_cv', 0.0), 3)} | "
            f"corr_w_active={_fmt_float(diagnostics.get('mean_weight_active_corr', 0.0), 3)} | "
            f"corr_w_fisher={_fmt_float(diagnostics.get('mean_weight_fisher_corr', 0.0), 3)} | "
            f"state_hh={_fmt_float(diagnostics.get('mean_state_hh_frac', 0.0), 2)} | "
            f"state_hl={_fmt_float(diagnostics.get('mean_state_hl_frac', 0.0), 2)} | "
            f"state_lh={_fmt_float(diagnostics.get('mean_state_lh_frac', 0.0), 2)} | "
            f"state_ll={_fmt_float(diagnostics.get('mean_state_ll_frac', 0.0), 2)}"
        )

        print_experts = bool(
            _cfg_get(wolf_cfg, "diagnostics_print_experts", False)
        )
        if not print_experts:
            return

        expert_diagnostics = diagnostics.get("expert_diagnostics", {})
        if not isinstance(expert_diagnostics, dict):
            return

        for expert_id, expert_diag in _sorted_expert_diagnostics(
            expert_diagnostics
        ):
            if not isinstance(expert_diag, dict):
                continue

            fallback = bool(expert_diag.get("fallback", False))
            top_client = expert_diag.get("top_client", None)
            top_client_text = "none" if top_client is None else str(top_client)

            rho_stats = expert_diag.get("rho_stats", {})
            gain_stats = expert_diag.get("kalman_gain_stats", {})
            support_stats = expert_diag.get("support_stats", {})
            abs_residual_stats = expert_diag.get("abs_residual_stats", {})
            mu_update_abs_stats = expert_diag.get("mu_update_abs_stats", {})

            print(
                f"{prefix}[Round {round_result.round_id:03d}][Expert {expert_id}] "
                f"valid={_safe_int(expert_diag.get('valid_clients', 0))} | "
                f"fallback={str(fallback).lower()} | "
                f"eff={_fmt_float(expert_diag.get('effective_clients', 0.0), 2)} | "
                f"entropy_norm={_fmt_float(expert_diag.get('weight_entropy_norm', 0.0), 3)} | "
                f"w_max={_fmt_float(expert_diag.get('weight_max', 0.0), 3)} | "
                f"top_client={top_client_text} | "
                f"top1_gap={_fmt_float(expert_diag.get('top1_gap', 0.0), 3)} | "
                f"rho={_fmt_float(rho_stats.get('mean', 0.0), 3)} | "
                f"rho_p10={_fmt_float(expert_diag.get('rho_p10', 0.0), 3)} | "
                f"K={_fmt_float(gain_stats.get('mean', 0.0), 3)} | "
                f"abs_resid={_fmt_float(abs_residual_stats.get('mean', 0.0), 3)} | "
                f"abs_resid_p90={_fmt_float(expert_diag.get('abs_residual_p90', 0.0), 3)} | "
                f"mu_update={_fmt_float(mu_update_abs_stats.get('mean', 0.0), 3)} | "
                f"cold_start={_fmt_float(expert_diag.get('cold_start_frac', 0.0), 3)} | "
                f"support={_fmt_float(support_stats.get('mean', 0.0), 3)} | "
                f"fisher_cv={_fmt_float(expert_diag.get('fisher_strength_cv', 0.0), 3)} | "
                f"active_cv={_fmt_float(expert_diag.get('active_count_cv', 0.0), 3)} | "
                f"corr_w_active={_fmt_float(expert_diag.get('weight_active_corr', 0.0), 3)} | "
                f"corr_w_fisher={_fmt_float(expert_diag.get('weight_fisher_corr', 0.0), 3)} | "
                f"hh={_fmt_float(expert_diag.get('state_hh_frac', 0.0), 2)} | "
                f"hl={_fmt_float(expert_diag.get('state_hl_frac', 0.0), 2)} | "
                f"lh={_fmt_float(expert_diag.get('state_lh_frac', 0.0), 2)} | "
                f"ll={_fmt_float(expert_diag.get('state_ll_frac', 0.0), 2)} | "
                f"zero_fisher={_safe_int(expert_diag.get('zero_fisher_clients', 0))} | "
                f"zero_active={_safe_int(expert_diag.get('zero_active_clients', 0))}"
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
    client_evidence_loaders: Optional[Sequence[DataLoader]] = None,
) -> FLServer:
    """
    构建 FLServer。

    train.py 后面可以直接调用这个函数。

    client_evidence_loaders:
        可选的客户端 evidence DataLoader 列表。
        当 expert_fisher.enabled=true 时必须传入，用于客户端训练完成后的
        expert K-FAC evidence 统计。
    """
    return FLServer(
        cfg=cfg,
        client_loaders=client_loaders,
        client_evidence_loaders=client_evidence_loaders,
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


def _sorted_expert_diagnostics(
    expert_diagnostics: Dict[Any, Any],
) -> List[tuple[int, Any]]:
    """
    按 expert_id 对 expert diagnostics 排序。

    兼容 int key 和 str key。
    """
    result: List[tuple[int, Any]] = []

    for raw_expert_id, expert_diag in expert_diagnostics.items():
        try:
            expert_id = int(raw_expert_id)
        except (TypeError, ValueError):
            continue

        result.append((expert_id, expert_diag))

    return sorted(result, key=lambda item: item[0])


def _fmt_float(
    value: Any,
    digits: int = 3,
) -> str:
    """
    把日志中的 float 格式化成固定小数位。

    NaN / Inf / 非数值统一显示为 nan。
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "nan"

    if not math.isfinite(value):
        return "nan"

    return f"{value:.{digits}f}"


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """
    安全转 int。
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


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