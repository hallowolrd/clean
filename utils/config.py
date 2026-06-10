from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

import yaml


# =========================
# 可扩展的合法取值注册区
# =========================
# 后续新增数据集、模型、聚合算法时，优先改这里。
SUPPORTED_DATASETS = {
    "cifar10",
    "cifar100",
}

SUPPORTED_MODELS = {
    "resnet_switch_moe",
    "resnet_sparse_moe_head",
}

SUPPORTED_AGG_METHODS = {
    "uniform",
    "sample_weighted",
    "fisher_only",
    "fisher_history_wolf",
}


class ConfigError(Exception):
    """配置相关错误。"""

    pass


class ConfigNode:
    """
    轻量级配置对象。

    支持两种读取方式：
        cfg.dataset
        cfg.agg.non_expert.method
        cfg.agg.expert.method

    也支持路径读取：
        cfg.get("agg.non_expert.method", "sample_weighted")
        cfg.get("agg.expert.method", "uniform")
    """

    def __init__(self, data: Mapping[str, Any]):
        for key, value in data.items():
            setattr(self, key, self._wrap(value))

    @staticmethod
    def _wrap(value: Any) -> Any:
        """把嵌套 dict 自动转成 ConfigNode。"""
        if isinstance(value, Mapping):
            return ConfigNode(value)

        if isinstance(value, list):
            return [
                ConfigNode._wrap(item)
                for item in value
            ]

        return value

    def get(self, path: str, default: Any = None) -> Any:
        """
        按路径读取配置。

        示例：
            cfg.get("agg.non_expert.method", "sample_weighted")
            cfg.get("agg.expert.method", "uniform")
            cfg.get("checkpoint.enabled", True)
        """
        current: Any = self

        for part in path.split("."):
            if isinstance(current, ConfigNode) and hasattr(current, part):
                current = getattr(current, part)
            else:
                return default

        return current

    def to_dict(self) -> Dict[str, Any]:
        """把 ConfigNode 递归转换回普通 dict。"""
        result = {}

        for key, value in self.__dict__.items():
            result[key] = self._unwrap(value)

        return result

    @staticmethod
    def _unwrap(value: Any) -> Any:
        """把 ConfigNode 递归转换成普通 Python 对象。"""
        if isinstance(value, ConfigNode):
            return value.to_dict()

        if isinstance(value, list):
            return [
                ConfigNode._unwrap(item)
                for item in value
            ]

        return value

    def __getitem__(self, key: str) -> Any:
        """支持 cfg["dataset"] 形式读取。"""
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def __repr__(self) -> str:
        return repr(self.to_dict())


def load_config(config_path: str | Path) -> ConfigNode:
    """
    读取配置文件，并返回 ConfigNode。

    主要流程：
    1. 读取 yaml
    2. 处理 include
    3. 合并默认值
    4. 自动生成 run_name / run_dir
    5. 做基础合法性检查
    6. 转成 ConfigNode
    """
    config_path = Path(config_path).expanduser().resolve()

    raw_cfg = _load_yaml_with_include(config_path)
    raw_cfg = _apply_defaults(raw_cfg)
    raw_cfg = _finalize_run_info(raw_cfg)
    _validate_config(raw_cfg)

    return ConfigNode(raw_cfg)


def save_config(
    cfg: ConfigNode | Mapping[str, Any],
    output_path: str | Path,
) -> None:
    """
    保存最终配置。

    一般用于保存：
        outputs/<run_name>/config_used.yaml
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(cfg, ConfigNode):
        cfg_dict = cfg.to_dict()
    else:
        cfg_dict = dict(cfg)

    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            cfg_dict,
            f,
            allow_unicode=True,
            sort_keys=False,
        )


def ensure_run_dir(cfg: ConfigNode | Mapping[str, Any]) -> Path:
    """
    创建实验输出目录，并返回 Path。

    注意：
        load_config 只负责生成 run_dir 字段；
        真正创建目录放在这里，避免读取配置时产生太多副作用。
    """
    if isinstance(cfg, ConfigNode):
        run_dir = Path(cfg.run_dir)
    else:
        run_dir = Path(cfg["run_dir"])

    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def _load_yaml_with_include(
    config_path: Path,
    stack: Optional[list[Path]] = None,
) -> Dict[str, Any]:
    """
    读取 yaml，并处理 include。

    支持：
        include: base.yaml

    也支持：
        include:
          - base.yaml
          - model/resnet.yaml

    子配置会覆盖 base 配置。
    """
    if stack is None:
        stack = []

    if config_path in stack:
        chain = " -> ".join(str(path) for path in stack + [config_path])
        raise ConfigError(f"检测到循环 include：{chain}")

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, Mapping):
        raise ConfigError(f"配置文件顶层必须是 dict：{config_path}")

    cfg = dict(cfg)
    include = cfg.pop("include", None)

    if include is None:
        return cfg

    if isinstance(include, str):
        include_files = [include]
    elif isinstance(include, list):
        include_files = include
    else:
        raise ConfigError("include 必须是字符串或字符串列表。")

    merged_cfg: Dict[str, Any] = {}

    for include_file in include_files:
        include_path = (config_path.parent / include_file).resolve()
        base_cfg = _load_yaml_with_include(
            include_path,
            stack=stack + [config_path],
        )
        merged_cfg = _deep_merge(merged_cfg, base_cfg)

    # 当前配置覆盖 include 进来的配置。
    merged_cfg = _deep_merge(merged_cfg, cfg)

    return merged_cfg


def _deep_merge(
    base: MutableMapping[str, Any],
    override: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    递归合并配置。

    规则：
    1. override 里的普通字段覆盖 base
    2. override 里的 dict 会递归覆盖 base 里的 dict
    3. list 不做递归合并，直接整体覆盖
    """
    result = copy.deepcopy(dict(base))

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def _apply_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    补齐默认配置。

    第一版只放最通用的默认值。
    后面新增模块时，也可以在这里继续补默认值。
    """
    cfg = copy.deepcopy(cfg)

    # 数据配置
    cfg.setdefault("dataset", "cifar10")
    cfg.setdefault("data_root", "./data")
    cfg.setdefault("num_classes", _infer_num_classes(cfg["dataset"]))
    cfg.setdefault("download_data", True)

    # 正常训练集是否使用随机数据增强。
    # Fisher / K-FAC evidence 数据集会单独强制关闭随机增强。
    cfg.setdefault("data_augmentation", True)

    # 联邦学习配置
    cfg.setdefault("num_clients", 10)
    cfg.setdefault("alpha", 0.1)
    cfg.setdefault("frac", 1.0)
    cfg.setdefault("rounds", 100)
    cfg.setdefault("local_epochs", 1)

    # dataloader 配置
    cfg.setdefault("batch_size", 64)
    cfg.setdefault("test_batch_size", 256)
    cfg.setdefault("num_workers", 2)
    cfg.setdefault("drop_last", False)
    cfg.setdefault("pin_memory", None)

    # evidence loader 的 worker 随机种子偏移。
    # 虽然 evidence_loader 默认 shuffle=False，但 worker 内部仍可能涉及随机性。
    cfg.setdefault("evidence_seed_offset", 100000)

    # 模型配置
    cfg.setdefault("model", "resnet_switch_moe")
    cfg.setdefault("num_experts", 4)
    cfg.setdefault("topk", 2)

    # 优化器配置
    cfg.setdefault("optimizer", {})
    cfg["optimizer"].setdefault("type", "sgd")
    cfg["optimizer"].setdefault("lr", 0.01)
    cfg["optimizer"].setdefault("momentum", 0.9)
    cfg["optimizer"].setdefault("weight_decay", 1e-4)

    # 聚合配置
    cfg.setdefault("agg", {})
    cfg["agg"].setdefault("non_expert", {})
    cfg["agg"]["non_expert"].setdefault("method", "sample_weighted")
    cfg["agg"].setdefault("expert", {})
    cfg["agg"]["expert"].setdefault("method", "uniform")

    # Expert Fisher / K-FAC evidence 配置。
    # 只有 expert_fisher.enabled=true 时，客户端本地训练完成后才会额外执行
    # evidence forward + backward 统计 expert K-FAC。
    cfg.setdefault("expert_fisher", {})
    cfg["expert_fisher"].setdefault("enabled", False)
    cfg["expert_fisher"].setdefault("model_mode", "eval")
    cfg["expert_fisher"].setdefault("include_bias", True)
    cfg["expert_fisher"].setdefault("loss_reduction", "sum")
    cfg["expert_fisher"].setdefault("max_batches", None)
    cfg["expert_fisher"].setdefault("min_active_count", 1)
    cfg["expert_fisher"].setdefault("min_valid_clients", 2)
    cfg["expert_fisher"].setdefault("fallback", "keep_global")
    cfg["expert_fisher"].setdefault("eps", 1.0e-8)

    # Fisher-only 诊断配置。
    # diagnostics_enabled:
    #     是否在 aggregation/fisher_only.py 中生成详细诊断字段。
    #
    # diagnostics_print:
    #     是否由 server.py 打印 Fisher 诊断日志到控制台 / train.log。
    #
    # diagnostics_print_every:
    #     每隔多少轮打印一次 Fisher 诊断。
    #
    # diagnostics_print_experts:
    #     是否逐 expert 打印诊断。第一版建议 false，避免日志太大。
    #
    # diagnostics_include_records:
    #     是否把完整 client-expert records / expert_weights 写入 summary.json。
    #     第一版建议 false，避免文件过大。
    #
    # diagnostics_prefix:
    #     Fisher 诊断日志前缀，方便 grep。
    cfg["expert_fisher"].setdefault("diagnostics_enabled", True)
    cfg["expert_fisher"].setdefault("diagnostics_print", True)
    cfg["expert_fisher"].setdefault("diagnostics_print_every", 1)
    cfg["expert_fisher"].setdefault("diagnostics_print_experts", False)
    cfg["expert_fisher"].setdefault("diagnostics_include_records", False)
    cfg["expert_fisher"].setdefault("diagnostics_prefix", "[FisherDiag]")

    # History-WoLF 服务端滤波配置。
    # 注意：
    # expert_fisher 负责“客户端如何采集 evidence”；
    # history_filter 负责“服务端如何用 evidence 维护 mu / P / age 并生成 expert 权重”。
    cfg.setdefault("history_filter", {})
    cfg["history_filter"].setdefault("global_warmup_rounds", 10)
    cfg["history_filter"].setdefault("post_warmup_force_filtered_weight", True)

    # active_count_min:
    #     判断 client-expert evidence 是否有效。
    #
    # min_valid_clients:
    #     判断 expert 参数本轮是否允许更新。
    #     这个参数不控制后台滤波器是否更新。
    #     例如 min_valid_clients=2 时，只有 1 个有效客户端：
    #         mu / P / age 仍然更新；
    #         expert 参数本轮保持上一轮全局值。
    cfg["history_filter"].setdefault("active_count_min", 1)
    cfg["history_filter"].setdefault("min_valid_clients", 2)

    # 判断能不能使用完整 History-WoLF 后台更新。
    # 历史不足时使用普通 Kalman fallback，但 post-warmup 后最终权重仍然来自 mu+。
    cfg["history_filter"].setdefault("history_warmup_age", 2)
    cfg["history_filter"].setdefault("min_history_clients", 3)
    cfg["history_filter"].setdefault("insufficient_history_update", "kalman_fallback")
    cfg["history_filter"].setdefault("warmup_weight_mode", "fisher_only")

    # log-score 空间 Kalman / WoLF 参数。
    cfg["history_filter"].setdefault("observation_R", 1.0)
    cfg["history_filter"].setdefault("mad_floor", 0.1)
    cfg["history_filter"].setdefault("init_P", 1.0)
    cfg["history_filter"].setdefault("process_noise_Q", 0.05)
    cfg["history_filter"].setdefault("robust_c", 2.0)
    cfg["history_filter"].setdefault("expert_weight_tau", 1.0)

    # 数值稳定参数。
    cfg["history_filter"].setdefault("w2_floor", 1.0e-6)
    cfg["history_filter"].setdefault("P_floor", 1.0e-8)
    cfg["history_filter"].setdefault("eps", 1.0e-12)

    # History-WoLF 诊断配置。
    cfg["history_filter"].setdefault("diagnostics_enabled", True)
    cfg["history_filter"].setdefault("diagnostics_print", True)
    cfg["history_filter"].setdefault("diagnostics_print_every", 1)
    cfg["history_filter"].setdefault("diagnostics_print_experts", False)
    cfg["history_filter"].setdefault("diagnostics_include_records", False)
    cfg["history_filter"].setdefault("diagnostics_prefix", "[HistWolfDiag]")

    # 运行配置
    cfg.setdefault("seed", 42)
    cfg.setdefault("device", "auto")
    cfg.setdefault("output_dir", "outputs")
    cfg.setdefault("run_name", "auto")

    # 输出目录命名策略
    cfg.setdefault("run", {})
    cfg["run"].setdefault("unique_name", True)
    cfg["run"].setdefault("overwrite", False)

    # 日志配置
    cfg.setdefault("logging", {})
    cfg["logging"].setdefault("log_every", 1)
    cfg["logging"].setdefault("save_config", True)
    cfg["logging"].setdefault("save_results_csv", True)

    # checkpoint 配置
    cfg.setdefault("checkpoint", {})
    cfg["checkpoint"].setdefault("enabled", True)
    cfg["checkpoint"].setdefault("save_latest", True)
    cfg["checkpoint"].setdefault("save_best", True)

    return cfg


def _finalize_run_info(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    生成 run_name 和 run_dir。

    规则：
    1. run_name 缺失 / 为空 / auto / null 时，自动根据实验设置生成
    2. 如果输出目录已存在，默认自动追加 _v2 / _v3
    3. 如果 run.overwrite=True，则允许使用已有目录，不自动追加版本号
    """
    cfg = copy.deepcopy(cfg)

    raw_run_name = cfg.get("run_name", "auto")
    should_auto_name = _is_auto_run_name(raw_run_name)

    if should_auto_name:
        run_name = _build_auto_run_name(cfg)
    else:
        run_name = _safe_name(raw_run_name)

    output_dir = Path(cfg.get("output_dir", "outputs"))
    unique_name = bool(cfg.get("run", {}).get("unique_name", True))
    overwrite = bool(cfg.get("run", {}).get("overwrite", False))

    if unique_name and not overwrite:
        run_name = _make_unique_run_name(run_name, output_dir)

    cfg["run_name"] = run_name
    cfg["run_dir"] = str(output_dir / run_name)

    return cfg


def _is_auto_run_name(value: Any) -> bool:
    """判断 run_name 是否需要自动生成。"""
    if value is None:
        return True

    text = str(value).strip().lower()

    return text in {
        "",
        "auto",
        "none",
        "null",
    }


def _build_auto_run_name(cfg: Mapping[str, Any]) -> str:
    """
    根据关键实验设置自动生成实验名。

    文件夹名只放关键字段，详细参数会保存到 config_used.yaml。
    """
    dataset = _safe_name(cfg.get("dataset", "dataset"))
    num_clients = _safe_name(cfg.get("num_clients", "c"))
    alpha = _safe_name(cfg.get("alpha", "iid"))
    model = _safe_name(cfg.get("model", "model"))
    num_experts = _safe_name(cfg.get("num_experts", "e"))
    topk = _safe_name(cfg.get("topk", "topk"))
    rounds = _safe_name(cfg.get("rounds", "r"))
    local_epochs = _safe_name(cfg.get("local_epochs", "ep"))
    seed = _safe_name(cfg.get("seed", "seed"))

    agg_cfg = cfg.get("agg", {})
    non_expert_method = _safe_name(
        agg_cfg.get("non_expert", {}).get("method", "non_expert")
    )
    expert_method = _safe_name(
        agg_cfg.get("expert", {}).get("method", "expert")
    )

    return (
        f"{dataset}"
        f"_c{num_clients}"
        f"_a{alpha}"
        f"_{model}"
        f"_e{num_experts}"
        f"_top{topk}"
        f"_r{rounds}"
        f"_ep{local_epochs}"
        f"_ne{non_expert_method}"
        f"_ex{expert_method}"
        f"_s{seed}"
    )


def _safe_name(value: Any) -> str:
    """
    把任意值转换成适合做文件夹名的字符串。

    示例：
        0.1 -> 0p1
        cuda:0 -> cuda_0
    """
    text = str(value).strip()
    text = text.replace(".", "p")
    text = text.replace("-", "m")
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text)

    return text.strip("_")


def _make_unique_run_name(
    run_name: str,
    output_dir: Path,
) -> str:
    """
    如果输出目录已存在，自动追加版本号。

    示例：
        exp
        exp_v2
        exp_v3
    """
    candidate = run_name
    index = 2

    while (output_dir / candidate).exists():
        candidate = f"{run_name}_v{index}"
        index += 1

    return candidate


def _infer_num_classes(dataset: str) -> int:
    """根据数据集名称推断类别数。"""
    if dataset == "cifar10":
        return 10

    if dataset == "cifar100":
        return 100

    # 对未知数据集先返回 10，真正合法性检查在 _validate_config 里做。
    return 10


def _validate_config(cfg: Mapping[str, Any]) -> None:
    """
    基础合法性检查。

    这个函数只检查通用配置和当前已接入的 expert_fisher / history_filter 配置。
    后面新增 history / Bayes 时，可以继续拆出新的 validate 函数。
    """
    dataset = cfg.get("dataset")
    if dataset not in SUPPORTED_DATASETS:
        raise ConfigError(
            f"不支持的数据集：{dataset}。"
            f"当前支持：{sorted(SUPPORTED_DATASETS)}"
        )

    model = cfg.get("model")
    if model not in SUPPORTED_MODELS:
        raise ConfigError(
            f"不支持的模型：{model}。"
            f"当前支持：{sorted(SUPPORTED_MODELS)}"
        )

    agg_cfg = cfg.get("agg", {})
    non_expert_method = agg_cfg.get("non_expert", {}).get("method")
    expert_method = agg_cfg.get("expert", {}).get("method")

    if non_expert_method not in SUPPORTED_AGG_METHODS:
        raise ConfigError(
            f"不支持的非专家参数聚合方法：{non_expert_method}。"
            f"当前支持：{sorted(SUPPORTED_AGG_METHODS)}"
        )

    if expert_method not in SUPPORTED_AGG_METHODS:
        raise ConfigError(
            f"不支持的专家参数聚合方法：{expert_method}。"
            f"当前支持：{sorted(SUPPORTED_AGG_METHODS)}"
        )

    # fisher_only / fisher_history_wolf 都是 expert-wise 权重聚合，
    # 只能用于 expert 参数组。
    # non_expert 参数仍然应该使用 uniform / sample_weighted。
    expert_only_methods = {
        "fisher_only",
        "fisher_history_wolf",
    }
    if non_expert_method in expert_only_methods:
        raise ConfigError(
            f"{non_expert_method} 只能用于 agg.expert.method，"
            "不能用于 agg.non_expert.method。"
        )

    _require_positive_int(cfg, "num_classes")
    _require_positive_int(cfg, "num_clients")
    _require_positive_int(cfg, "rounds")
    _require_positive_int(cfg, "local_epochs")
    _require_positive_int(cfg, "batch_size")
    _require_positive_int(cfg, "test_batch_size")
    _require_non_negative_int(cfg, "num_workers")
    _require_positive_int(cfg, "num_experts")
    _require_positive_int(cfg, "topk")

    if int(cfg["topk"]) > int(cfg["num_experts"]):
        raise ConfigError(
            f"topk 不能大于 num_experts："
            f"topk={cfg['topk']}, num_experts={cfg['num_experts']}"
        )

    frac = float(cfg.get("frac"))
    if not (0.0 < frac <= 1.0):
        raise ConfigError(f"frac 必须在 (0, 1] 范围内，当前值：{frac}")

    alpha = float(cfg.get("alpha"))
    if alpha <= 0:
        raise ConfigError(f"alpha 必须大于 0，当前值：{alpha}")

    # 优化器参数必须统一写在 optimizer 下，避免同时存在两套配置入口。
    forbidden_top_level_optimizer_keys = {
        "lr",
        "momentum",
        "weight_decay",
    }
    for key in forbidden_top_level_optimizer_keys:
        if key in cfg:
            raise ConfigError(
                f"请不要在顶层配置 {key}。"
                f"请统一写到 optimizer.{key} 下面。"
            )

    optimizer_cfg = cfg.get("optimizer", {})
    optimizer_type = optimizer_cfg.get("type")
    if optimizer_type not in {"sgd", "adam", "adamw"}:
        raise ConfigError(
            f"不支持的优化器：{optimizer_type}。"
            f"当前支持：sgd, adam, adamw"
        )

    lr = float(optimizer_cfg.get("lr"))
    if lr <= 0:
        raise ConfigError(f"optimizer.lr 必须大于 0，当前值：{lr}")

    weight_decay = float(optimizer_cfg.get("weight_decay"))
    if weight_decay < 0:
        raise ConfigError(
            f"optimizer.weight_decay 不能小于 0，当前值：{weight_decay}"
        )

    _validate_expert_fisher_config(
        cfg=cfg,
        expert_method=expert_method,
    )
    _validate_history_filter_config(
        cfg=cfg,
        expert_method=expert_method,
    )


def _validate_expert_fisher_config(
    cfg: Mapping[str, Any],
    expert_method: str,
) -> None:
    """
    检查 expert_fisher 配置。

    expert_fisher 用于：
    1. 客户端本地训练后额外做 evidence forward + backward
    2. 只对 expert Linear 层统计 K-FAC A / B
    3. 给 fisher_only / fisher_history_wolf 聚合器提供 active_count / mean_A / mean_B / score
    4. 控制 fisher_only 诊断字段生成和日志打印
    """
    expert_fisher_cfg = cfg.get("expert_fisher", {})

    if not isinstance(expert_fisher_cfg, Mapping):
        raise ConfigError("expert_fisher 必须是一个 dict。")

    enabled = bool(expert_fisher_cfg.get("enabled", False))
    if expert_method in {"fisher_only", "fisher_history_wolf"} and not enabled:
        raise ConfigError(
            f"agg.expert.method={expert_method} 时，必须设置 "
            "expert_fisher.enabled=true。"
        )

    model_mode = str(expert_fisher_cfg.get("model_mode", "eval")).lower()
    if model_mode not in {"eval", "train"}:
        raise ConfigError(
            f"expert_fisher.model_mode 只支持 eval / train，当前值：{model_mode}"
        )

    loss_reduction = str(expert_fisher_cfg.get("loss_reduction", "sum")).lower()
    if loss_reduction not in {"sum", "mean"}:
        raise ConfigError(
            "expert_fisher.loss_reduction 只支持 sum / mean，"
            f"当前值：{loss_reduction}"
        )

    fallback = str(expert_fisher_cfg.get("fallback", "keep_global")).lower()
    if fallback not in {"keep_global"}:
        raise ConfigError(
            "expert_fisher.fallback 当前只支持 keep_global，"
            f"当前值：{fallback}"
        )

    min_active_count = expert_fisher_cfg.get("min_active_count", 1)
    if not isinstance(min_active_count, int) or isinstance(min_active_count, bool):
        raise ConfigError(
            "expert_fisher.min_active_count 必须是非负整数，"
            f"当前值：{min_active_count}"
        )
    if min_active_count < 0:
        raise ConfigError(
            "expert_fisher.min_active_count 必须是非负整数，"
            f"当前值：{min_active_count}"
        )

    min_valid_clients = expert_fisher_cfg.get("min_valid_clients", 2)
    if (
        not isinstance(min_valid_clients, int)
        or isinstance(min_valid_clients, bool)
        or min_valid_clients <= 0
    ):
        raise ConfigError(
            "expert_fisher.min_valid_clients 必须是正整数，"
            f"当前值：{min_valid_clients}"
        )

    max_batches = expert_fisher_cfg.get("max_batches", None)
    if max_batches is not None:
        if (
            not isinstance(max_batches, int)
            or isinstance(max_batches, bool)
            or max_batches <= 0
        ):
            raise ConfigError(
                "expert_fisher.max_batches 如果不为 null，必须是正整数，"
                f"当前值：{max_batches}"
            )

    eps = float(expert_fisher_cfg.get("eps", 1.0e-8))
    if eps <= 0:
        raise ConfigError(f"expert_fisher.eps 必须大于 0，当前值：{eps}")

    _validate_expert_fisher_diagnostics_config(expert_fisher_cfg)


def _validate_expert_fisher_diagnostics_config(
    expert_fisher_cfg: Mapping[str, Any],
) -> None:
    """
    检查 Fisher-only 诊断配置。

    这些字段主要服务于：
    1. aggregation/fisher_only.py 生成 diagnostics
    2. fl/server.py 打印 [FisherDiag] 日志
    """
    bool_fields = [
        "diagnostics_enabled",
        "diagnostics_print",
        "diagnostics_print_experts",
        "diagnostics_include_records",
    ]

    for key in bool_fields:
        value = expert_fisher_cfg.get(key)
        if not isinstance(value, bool):
            raise ConfigError(
                f"expert_fisher.{key} 必须是 bool，当前值：{value}"
            )

    diagnostics_print_every = expert_fisher_cfg.get("diagnostics_print_every", 1)
    if (
        not isinstance(diagnostics_print_every, int)
        or isinstance(diagnostics_print_every, bool)
        or diagnostics_print_every <= 0
    ):
        raise ConfigError(
            "expert_fisher.diagnostics_print_every 必须是正整数，"
            f"当前值：{diagnostics_print_every}"
        )

    diagnostics_prefix = expert_fisher_cfg.get("diagnostics_prefix", "[FisherDiag]")
    if not isinstance(diagnostics_prefix, str) or len(diagnostics_prefix.strip()) == 0:
        raise ConfigError(
            "expert_fisher.diagnostics_prefix 必须是非空字符串，"
            f"当前值：{diagnostics_prefix}"
        )


def _validate_history_filter_config(
    cfg: Mapping[str, Any],
    expert_method: str,
) -> None:
    """
    检查 history_filter 配置。

    history_filter 用于 fisher_history_wolf：
    1. 控制 warmup 阶段最终权重来源；
    2. 控制 client-expert 级别 mu / P / age 后台滤波更新；
    3. 控制 History-WoLF 诊断字段生成和日志打印。
    """
    history_filter_cfg = cfg.get("history_filter", {})

    if not isinstance(history_filter_cfg, Mapping):
        raise ConfigError("history_filter 必须是一个 dict。")

    # 这些字段即使当前不启用 fisher_history_wolf，也做轻量检查。
    # 这样配置写错时可以尽早暴露。
    _require_non_negative_int(history_filter_cfg, "global_warmup_rounds")
    _require_non_negative_int(history_filter_cfg, "active_count_min")
    _require_positive_int(history_filter_cfg, "min_valid_clients")
    _require_non_negative_int(history_filter_cfg, "history_warmup_age")
    _require_positive_int(history_filter_cfg, "min_history_clients")

    post_warmup_force_filtered_weight = history_filter_cfg.get(
        "post_warmup_force_filtered_weight",
        True,
    )
    if not isinstance(post_warmup_force_filtered_weight, bool):
        raise ConfigError(
            "history_filter.post_warmup_force_filtered_weight 必须是 bool，"
            f"当前值：{post_warmup_force_filtered_weight}"
        )

    insufficient_history_update = str(
        history_filter_cfg.get("insufficient_history_update", "kalman_fallback")
    ).lower()
    if insufficient_history_update not in {"kalman_fallback"}:
        raise ConfigError(
            "history_filter.insufficient_history_update 当前只支持 kalman_fallback，"
            f"当前值：{insufficient_history_update}"
        )

    warmup_weight_mode = str(
        history_filter_cfg.get("warmup_weight_mode", "fisher_only")
    ).lower()
    if warmup_weight_mode not in {"fisher_only"}:
        raise ConfigError(
            "history_filter.warmup_weight_mode 当前只支持 fisher_only，"
            f"当前值：{warmup_weight_mode}"
        )

    positive_float_fields = [
        "observation_R",
        "mad_floor",
        "init_P",
        "expert_weight_tau",
        "w2_floor",
        "P_floor",
        "eps",
    ]
    for key in positive_float_fields:
        value = float(history_filter_cfg.get(key))
        if value <= 0:
            raise ConfigError(
                f"history_filter.{key} 必须大于 0，当前值：{value}"
            )

    process_noise_Q = float(history_filter_cfg.get("process_noise_Q"))
    if process_noise_Q < 0:
        raise ConfigError(
            "history_filter.process_noise_Q 必须大于等于 0，"
            f"当前值：{process_noise_Q}"
        )

    robust_c = float(history_filter_cfg.get("robust_c"))
    if robust_c <= 0:
        raise ConfigError(
            f"history_filter.robust_c 必须大于 0，当前值：{robust_c}"
        )

    # fisher_history_wolf 才真正依赖 history_filter。
    # 这里保留明确检查，防止后面有人误删默认配置。
    if expert_method == "fisher_history_wolf":
        required_keys = [
            "global_warmup_rounds",
            "active_count_min",
            "min_valid_clients",
            "history_warmup_age",
            "min_history_clients",
            "observation_R",
            "mad_floor",
            "init_P",
            "process_noise_Q",
            "robust_c",
            "expert_weight_tau",
            "w2_floor",
            "P_floor",
            "eps",
        ]
        for key in required_keys:
            if key not in history_filter_cfg:
                raise ConfigError(
                    f"agg.expert.method=fisher_history_wolf 时，"
                    f"history_filter.{key} 不能为空。"
                )

    _validate_history_filter_diagnostics_config(history_filter_cfg)


def _validate_history_filter_diagnostics_config(
    history_filter_cfg: Mapping[str, Any],
) -> None:
    """
    检查 History-WoLF 诊断配置。

    这些字段主要服务于：
    1. aggregation/fisher_history_wolf.py 生成 diagnostics
    2. fl/server.py 打印 [HistWolfDiag] 日志
    """
    bool_fields = [
        "diagnostics_enabled",
        "diagnostics_print",
        "diagnostics_print_experts",
        "diagnostics_include_records",
    ]

    for key in bool_fields:
        value = history_filter_cfg.get(key)
        if not isinstance(value, bool):
            raise ConfigError(
                f"history_filter.{key} 必须是 bool，当前值：{value}"
            )

    diagnostics_print_every = history_filter_cfg.get("diagnostics_print_every", 1)
    if (
        not isinstance(diagnostics_print_every, int)
        or isinstance(diagnostics_print_every, bool)
        or diagnostics_print_every <= 0
    ):
        raise ConfigError(
            "history_filter.diagnostics_print_every 必须是正整数，"
            f"当前值：{diagnostics_print_every}"
        )

    diagnostics_prefix = history_filter_cfg.get("diagnostics_prefix", "[HistWolfDiag]")
    if not isinstance(diagnostics_prefix, str) or len(diagnostics_prefix.strip()) == 0:
        raise ConfigError(
            "history_filter.diagnostics_prefix 必须是非空字符串，"
            f"当前值：{diagnostics_prefix}"
        )


def _require_positive_int(
    cfg: Mapping[str, Any],
    key: str,
) -> None:
    """检查某个字段是否为正整数。"""
    value = cfg.get(key)

    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{key} 必须是正整数，当前值：{value}")


def _require_non_negative_int(
    cfg: Mapping[str, Any],
    key: str,
) -> None:
    """检查某个字段是否为非负整数。"""
    value = cfg.get(key)

    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigError(f"{key} 必须是非负整数，当前值：{value}")