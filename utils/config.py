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

    # 新增：纯 FL ResNet18 FedAvg baseline。
    # 这个模型没有 MoE、没有 expert、没有 router。
    "resnet18_fedavg",
}

SUPPORTED_AGG_METHODS = {
    "uniform",
    "sample_weighted",

    # 原 FL+MoE expert-wise 聚合方法。
    "fisher_only",
    "fisher_history_wolf",

    # 新增 pure-FL / full-model / client-wise 聚合方法。
    "fisher_only_global",
    "fisher_history_wolf_global",
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
        cfg.get("checkpoint.enabled", True)
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
            return [ConfigNode._wrap(item) for item in value]

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
            return [ConfigNode._unwrap(item) for item in value]

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

    # =========================
    # 数据配置
    # =========================
    cfg.setdefault("dataset", "cifar10")
    cfg.setdefault("data_root", "./data")
    cfg.setdefault("num_classes", _infer_num_classes(cfg["dataset"]))
    cfg.setdefault("download_data", True)

    # 正常训练集是否使用随机数据增强。
    # Fisher / K-FAC evidence 数据集会单独强制关闭随机增强。
    cfg.setdefault("data_augmentation", True)

    # =========================
    # 联邦学习配置
    # =========================
    cfg.setdefault("num_clients", 10)
    cfg.setdefault("alpha", 0.1)
    cfg.setdefault("frac", 1.0)
    cfg.setdefault("rounds", 100)
    cfg.setdefault("local_epochs", 1)

    # =========================
    # DataLoader 配置
    # =========================
    cfg.setdefault("batch_size", 64)
    cfg.setdefault("test_batch_size", 256)
    cfg.setdefault("num_workers", 2)
    cfg.setdefault("drop_last", False)
    cfg.setdefault("pin_memory", None)

    # evidence loader 的 worker 随机种子偏移。
    # 虽然 evidence_loader 默认 shuffle=False，但 worker 内部仍可能涉及随机性。
    cfg.setdefault("evidence_seed_offset", 100000)

    # =========================
    # 模型配置
    # =========================
    cfg.setdefault("model", "resnet_switch_moe")

    # MoE 模型使用 num_experts / topk。
    # pure-FL 模型 resnet18_fedavg 不使用这两个字段，
    # 但保留默认值可以兼容 base.yaml 和 run_name 自动生成逻辑。
    cfg.setdefault("num_experts", 4)
    cfg.setdefault("topk", 2)

    # resnet18_fedavg 默认适配 CIFAR 的 3x32x32 输入。
    cfg.setdefault("input_shape", [3, 32, 32])

    # =========================
    # 优化器配置
    # =========================
    cfg.setdefault("optimizer", {})
    cfg["optimizer"].setdefault("type", "sgd")
    cfg["optimizer"].setdefault("lr", 0.01)
    cfg["optimizer"].setdefault("momentum", 0.9)
    cfg["optimizer"].setdefault("weight_decay", 1e-4)

    # =========================
    # 聚合配置
    # =========================
    cfg.setdefault("agg", {})
    cfg["agg"].setdefault("non_expert", {})
    cfg["agg"]["non_expert"].setdefault("method", "sample_weighted")
    cfg["agg"].setdefault("expert", {})
    cfg["agg"]["expert"].setdefault("method", "uniform")

    # =========================
    # Expert Fisher / K-FAC evidence 配置
    # =========================
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
    cfg["expert_fisher"].setdefault("diagnostics_enabled", True)
    cfg["expert_fisher"].setdefault("diagnostics_print", True)
    cfg["expert_fisher"].setdefault("diagnostics_print_every", 1)
    cfg["expert_fisher"].setdefault("diagnostics_print_experts", False)
    cfg["expert_fisher"].setdefault("diagnostics_include_records", False)
    cfg["expert_fisher"].setdefault("diagnostics_prefix", "[FisherDiag]")

    # =========================
    # Full-model Fisher evidence 配置
    # =========================
    # 这是新增的 pure-FL / full-model / client-wise Fisher evidence。
    #
    # 和 expert_fisher 的区别：
    #   expert_fisher:
    #       统计 MoE expert K-FAC，写入 update.extra["expert_kfac"]。
    #
    #   full_model_fisher:
    #       统计整模型 mean(grad^2)，写入 update.extra["global_fisher"]。
    #
    # 只有 fisher_only_global / fisher_history_wolf_global 才需要开启。
    cfg.setdefault("full_model_fisher", {})
    cfg["full_model_fisher"].setdefault("enabled", False)
    cfg["full_model_fisher"].setdefault("model_mode", "eval")
    cfg["full_model_fisher"].setdefault("max_batches", 10)
    cfg["full_model_fisher"].setdefault("eps", 1.0e-8)
    cfg["full_model_fisher"].setdefault("min_valid_clients", 2)
    cfg["full_model_fisher"].setdefault("missing_policy", "error")
    cfg["full_model_fisher"].setdefault("diagnostics_enabled", True)
    cfg["full_model_fisher"].setdefault("diagnostics_print", True)
    cfg["full_model_fisher"].setdefault("diagnostics_print_every", 1)
    cfg["full_model_fisher"].setdefault("diagnostics_include_records", False)
    cfg["full_model_fisher"].setdefault("diagnostics_prefix", "[FullFisherDiag]")

    # =========================
    # Fisher-only global 聚合器配置
    # =========================
    # fisher_only_global 第一版故意保持简单：
    #   score_i = num_samples_i * fisher_strength_i
    #
    # 不加温度系数、不加 uniform mix、不加额外归一化。
    cfg.setdefault("fisher_only_global", {})
    cfg["fisher_only_global"].setdefault("enabled", False)

    # =========================
    # Fisher-History-WoLF 服务端历史滤波聚合配置
    # =========================
    # 注意：
    #   expert_fisher 负责客户端 evidence 采集；
    #   fisher_history_wolf 负责服务端如何把 evidence 转成 expert 聚合权重。
    cfg.setdefault("fisher_history_wolf", {})
    cfg["fisher_history_wolf"].setdefault("init_P", 1.0)
    cfg["fisher_history_wolf"].setdefault("process_noise_Q", 0.05)
    cfg["fisher_history_wolf"].setdefault("observation_R", 1.0)
    cfg["fisher_history_wolf"].setdefault("robust_c", 2.0)
    cfg["fisher_history_wolf"].setdefault("eps", 1.0e-8)
    cfg["fisher_history_wolf"].setdefault("diagnostics_enabled", True)
    cfg["fisher_history_wolf"].setdefault("diagnostics_print", True)
    cfg["fisher_history_wolf"].setdefault("diagnostics_print_every", 1)
    cfg["fisher_history_wolf"].setdefault("diagnostics_print_experts", False)
    cfg["fisher_history_wolf"].setdefault("diagnostics_include_records", False)
    cfg["fisher_history_wolf"].setdefault("diagnostics_prefix", "[FisherWolfDiag]")

    # =========================
    # Fisher-History-WoLF global 聚合器配置
    # =========================
    # 这是新增的 pure-FL / full-model / client-wise 历史滤波器。
    #
    # 和 fisher_history_wolf 的区别：
    #   fisher_history_wolf:
    #       每个 (client_id, expert_id) 维护历史状态。
    #
    #   fisher_history_wolf_global:
    #       每个 client_id 维护一个历史状态。
    cfg.setdefault("fisher_history_wolf_global", {})
    cfg["fisher_history_wolf_global"].setdefault("enabled", False)
    cfg["fisher_history_wolf_global"].setdefault("init_P", 1.0)
    cfg["fisher_history_wolf_global"].setdefault("process_noise_Q", 0.05)
    cfg["fisher_history_wolf_global"].setdefault("observation_R", 1.0)
    cfg["fisher_history_wolf_global"].setdefault("robust_c", 2.0)
    cfg["fisher_history_wolf_global"].setdefault("eps", 1.0e-8)
    cfg["fisher_history_wolf_global"].setdefault("diagnostics_enabled", True)
    cfg["fisher_history_wolf_global"].setdefault("diagnostics_print", True)
    cfg["fisher_history_wolf_global"].setdefault("diagnostics_print_every", 1)
    cfg["fisher_history_wolf_global"].setdefault("diagnostics_include_records", False)
    cfg["fisher_history_wolf_global"].setdefault(
        "diagnostics_prefix",
        "[FullFisherWoLFDiag]",
    )

    # =========================
    # 运行配置
    # =========================
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

    这个函数只检查通用配置和当前已接入的：
        - expert_fisher
        - fisher_history_wolf
        - full_model_fisher
        - fisher_only_global
        - fisher_history_wolf_global

    后面新增 Bayes 等模块时，可以继续拆出新的 validate 函数。
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
    # 只能用于 expert 参数组，不能用于 non_expert。
    if non_expert_method in {"fisher_only", "fisher_history_wolf"}:
        raise ConfigError(
            "fisher_only / fisher_history_wolf 只能用于 agg.expert.method，"
            "不能用于 agg.non_expert.method。"
            "如果你是在 pure-FL 整模型场景使用 Fisher，"
            "请改用 fisher_only_global / fisher_history_wolf_global。"
        )

    # fisher_only_global / fisher_history_wolf_global 是 pure-FL 整模型聚合，
    # 只能用于 non_expert 参数组，不能用于 expert。
    if expert_method in {"fisher_only_global", "fisher_history_wolf_global"}:
        raise ConfigError(
            "fisher_only_global / fisher_history_wolf_global 只能用于 "
            "agg.non_expert.method，不能用于 agg.expert.method。"
        )

    if expert_method in {"fisher_only", "fisher_history_wolf"}:
        if model == "resnet18_fedavg":
            raise ConfigError(
                "resnet18_fedavg 是 pure-FL 模型，没有 expert 参数，"
                "不能使用 expert-wise fisher_only / fisher_history_wolf。"
            )

    if non_expert_method in {"fisher_only_global", "fisher_history_wolf_global"}:
        if model != "resnet18_fedavg":
            raise ConfigError(
                "fisher_only_global / fisher_history_wolf_global 是为 pure-FL "
                "整模型聚合准备的。当前建议只和 model=resnet18_fedavg 一起使用。"
            )

    _require_positive_int(cfg, "num_classes")
    _require_positive_int(cfg, "num_clients")
    _require_positive_int(cfg, "rounds")
    _require_positive_int(cfg, "local_epochs")
    _require_positive_int(cfg, "batch_size")
    _require_positive_int(cfg, "test_batch_size")
    _require_non_negative_int(cfg, "num_workers")

    # pure-FL 模型不使用 num_experts / topk，但为了兼容 base.yaml 和 run_name，
    # 仍然要求它们是正整数。这样不会影响 resnet18_fedavg 的实际模型结构。
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

    _validate_input_shape_config(cfg)

    _validate_expert_fisher_config(
        cfg=cfg,
        expert_method=expert_method,
    )
    _validate_fisher_history_wolf_config(
        cfg=cfg,
        expert_method=expert_method,
    )
    _validate_full_model_fisher_config(
        cfg=cfg,
        non_expert_method=non_expert_method,
    )
    _validate_fisher_only_global_config(
        cfg=cfg,
        non_expert_method=non_expert_method,
    )
    _validate_fisher_history_wolf_global_config(
        cfg=cfg,
        non_expert_method=non_expert_method,
    )


def _validate_input_shape_config(cfg: Mapping[str, Any]) -> None:
    """
    检查 input_shape 配置。

    当前主要服务于 resnet18_fedavg：
        input_shape: [3, 32, 32]
    """
    input_shape = cfg.get("input_shape", [3, 32, 32])

    if input_shape is None:
        return

    if not isinstance(input_shape, list) or len(input_shape) != 3:
        raise ConfigError(
            "input_shape 必须是长度为 3 的列表，例如 [3, 32, 32]，"
            f"当前值：{input_shape}"
        )

    for value in input_shape:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigError(
                "input_shape 中每个元素都必须是正整数，"
                f"当前值：{input_shape}"
            )

    model = cfg.get("model")

    if model == "resnet18_fedavg":
        if input_shape != [3, 32, 32]:
            raise ConfigError(
                "resnet18_fedavg 当前保留原始 fedavg.py 的结构，"
                "只支持 input_shape: [3, 32, 32]。"
                f"当前值：{input_shape}"
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
        3. 给 fisher_only / fisher_history_wolf 聚合器提供
           active_count / mean_A / mean_B / fisher_strength / score
        4. 控制 fisher_only 诊断字段生成和日志打印
    """
    expert_fisher_cfg = cfg.get("expert_fisher", {})

    if not isinstance(expert_fisher_cfg, Mapping):
        raise ConfigError("expert_fisher 必须是一个 dict。")

    enabled = bool(expert_fisher_cfg.get("enabled", False))

    if expert_method in {"fisher_only", "fisher_history_wolf"} and not enabled:
        raise ConfigError(
            "agg.expert.method=fisher_only 或 fisher_history_wolf 时，必须设置 "
            "expert_fisher.enabled=true。"
        )

    model_mode = str(expert_fisher_cfg.get("model_mode", "eval")).lower()
    if model_mode not in {"eval", "train"}:
        raise ConfigError(
            f"expert_fisher.model_mode 只支持 eval / train，当前值：{model_mode}"
        )

    loss_reduction = str(
        expert_fisher_cfg.get("loss_reduction", "sum")
    ).lower()
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

    diagnostics_print_every = expert_fisher_cfg.get(
        "diagnostics_print_every",
        1,
    )
    if (
        not isinstance(diagnostics_print_every, int)
        or isinstance(diagnostics_print_every, bool)
        or diagnostics_print_every <= 0
    ):
        raise ConfigError(
            "expert_fisher.diagnostics_print_every 必须是正整数，"
            f"当前值：{diagnostics_print_every}"
        )

    diagnostics_prefix = expert_fisher_cfg.get(
        "diagnostics_prefix",
        "[FisherDiag]",
    )
    if not isinstance(diagnostics_prefix, str) or len(diagnostics_prefix.strip()) == 0:
        raise ConfigError(
            "expert_fisher.diagnostics_prefix 必须是非空字符串，"
            f"当前值：{diagnostics_prefix}"
        )


def _validate_fisher_history_wolf_config(
    cfg: Mapping[str, Any],
    expert_method: str,
) -> None:
    """
    检查 fisher_history_wolf 配置。

    fisher_history_wolf 只负责服务端历史滤波聚合：
        1. 用 fisher_strength 构造 normalized log Fisher observation
        2. 用 WoLF-IMQ residual 降低异常观测对历史状态的影响
        3. 用 active_count 只做 support confidence，不做高 usage 奖励
        4. 最终生成 expert-wise 客户端权重
    """
    wolf_cfg = cfg.get("fisher_history_wolf", {})

    if not isinstance(wolf_cfg, Mapping):
        raise ConfigError("fisher_history_wolf 必须是一个 dict。")

    init_P = float(wolf_cfg.get("init_P", 1.0))
    if init_P <= 0:
        raise ConfigError(
            f"fisher_history_wolf.init_P 必须大于 0，当前值：{init_P}"
        )

    process_noise_Q = float(wolf_cfg.get("process_noise_Q", 0.05))
    if process_noise_Q < 0:
        raise ConfigError(
            "fisher_history_wolf.process_noise_Q 不能小于 0，"
            f"当前值：{process_noise_Q}"
        )

    observation_R = float(wolf_cfg.get("observation_R", 1.0))
    if observation_R <= 0:
        raise ConfigError(
            "fisher_history_wolf.observation_R 必须大于 0，"
            f"当前值：{observation_R}"
        )

    robust_c = float(wolf_cfg.get("robust_c", 2.0))
    if robust_c <= 0:
        raise ConfigError(
            f"fisher_history_wolf.robust_c 必须大于 0，当前值：{robust_c}"
        )

    eps = float(wolf_cfg.get("eps", 1.0e-8))
    if eps <= 0:
        raise ConfigError(
            f"fisher_history_wolf.eps 必须大于 0，当前值：{eps}"
        )

    _validate_fisher_history_wolf_diagnostics_config(wolf_cfg)

    if expert_method == "fisher_history_wolf":
        # 这里不做额外逻辑，只保留位置方便以后扩展。
        # expert_fisher.enabled=true 的强约束已经在 _validate_expert_fisher_config 中完成。
        return


def _validate_fisher_history_wolf_diagnostics_config(
    wolf_cfg: Mapping[str, Any],
) -> None:
    """
    检查 Fisher-History-WoLF 诊断配置。

    这些字段主要服务于：
        1. aggregation/fisher_history_wolf.py 生成 diagnostics
        2. fl/server.py 打印 [FisherWolfDiag] 日志
    """
    bool_fields = [
        "diagnostics_enabled",
        "diagnostics_print",
        "diagnostics_print_experts",
        "diagnostics_include_records",
    ]

    for key in bool_fields:
        value = wolf_cfg.get(key)
        if not isinstance(value, bool):
            raise ConfigError(
                f"fisher_history_wolf.{key} 必须是 bool，当前值：{value}"
            )

    diagnostics_print_every = wolf_cfg.get(
        "diagnostics_print_every",
        1,
    )
    if (
        not isinstance(diagnostics_print_every, int)
        or isinstance(diagnostics_print_every, bool)
        or diagnostics_print_every <= 0
    ):
        raise ConfigError(
            "fisher_history_wolf.diagnostics_print_every 必须是正整数，"
            f"当前值：{diagnostics_print_every}"
        )

    diagnostics_prefix = wolf_cfg.get(
        "diagnostics_prefix",
        "[FisherWolfDiag]",
    )
    if not isinstance(diagnostics_prefix, str) or len(diagnostics_prefix.strip()) == 0:
        raise ConfigError(
            "fisher_history_wolf.diagnostics_prefix 必须是非空字符串，"
            f"当前值：{diagnostics_prefix}"
        )


def _validate_full_model_fisher_config(
    cfg: Mapping[str, Any],
    non_expert_method: str,
) -> None:
    """
    检查 full_model_fisher 配置。

    full_model_fisher 用于 pure-FL：
        1. 客户端本地训练完成后，额外跑 evidence pass；
        2. 对整个模型统计 mean(grad^2)；
        3. 写入 update.extra["global_fisher"]；
        4. 给 fisher_only_global / fisher_history_wolf_global 使用。
    """
    fisher_cfg = cfg.get("full_model_fisher", {})

    if not isinstance(fisher_cfg, Mapping):
        raise ConfigError("full_model_fisher 必须是一个 dict。")

    enabled = bool(fisher_cfg.get("enabled", False))

    if non_expert_method in {"fisher_only_global", "fisher_history_wolf_global"}:
        if not enabled:
            raise ConfigError(
                "agg.non_expert.method=fisher_only_global 或 "
                "fisher_history_wolf_global 时，必须设置 "
                "full_model_fisher.enabled=true。"
            )

    model_mode = str(fisher_cfg.get("model_mode", "eval")).lower()
    if model_mode not in {"eval", "train"}:
        raise ConfigError(
            f"full_model_fisher.model_mode 只支持 eval / train，当前值：{model_mode}"
        )

    max_batches = fisher_cfg.get("max_batches", 10)
    if max_batches is not None:
        if (
            not isinstance(max_batches, int)
            or isinstance(max_batches, bool)
            or max_batches <= 0
        ):
            raise ConfigError(
                "full_model_fisher.max_batches 如果不为 null，必须是正整数，"
                f"当前值：{max_batches}"
            )

    eps = float(fisher_cfg.get("eps", 1.0e-8))
    if eps <= 0:
        raise ConfigError(f"full_model_fisher.eps 必须大于 0，当前值：{eps}")

    min_valid_clients = fisher_cfg.get("min_valid_clients", 2)
    if (
        not isinstance(min_valid_clients, int)
        or isinstance(min_valid_clients, bool)
        or min_valid_clients <= 0
    ):
        raise ConfigError(
            "full_model_fisher.min_valid_clients 必须是正整数，"
            f"当前值：{min_valid_clients}"
        )

    missing_policy = str(fisher_cfg.get("missing_policy", "error")).lower()
    if missing_policy not in {"error", "skip", "uniform"}:
        raise ConfigError(
            "full_model_fisher.missing_policy 只支持 error / skip / uniform，"
            f"当前值：{missing_policy}"
        )

    _validate_full_model_fisher_diagnostics_config(fisher_cfg)


def _validate_full_model_fisher_diagnostics_config(
    fisher_cfg: Mapping[str, Any],
) -> None:
    """
    检查 full_model_fisher 诊断配置。

    这些字段主要服务于：
        1. aggregation/fisher_only_global.py 生成 diagnostics
        2. fl/server.py 打印 [FullFisherDiag] 日志
    """
    bool_fields = [
        "diagnostics_enabled",
        "diagnostics_print",
        "diagnostics_include_records",
    ]

    for key in bool_fields:
        value = fisher_cfg.get(key)
        if not isinstance(value, bool):
            raise ConfigError(
                f"full_model_fisher.{key} 必须是 bool，当前值：{value}"
            )

    diagnostics_print_every = fisher_cfg.get(
        "diagnostics_print_every",
        1,
    )
    if (
        not isinstance(diagnostics_print_every, int)
        or isinstance(diagnostics_print_every, bool)
        or diagnostics_print_every <= 0
    ):
        raise ConfigError(
            "full_model_fisher.diagnostics_print_every 必须是正整数，"
            f"当前值：{diagnostics_print_every}"
        )

    diagnostics_prefix = fisher_cfg.get(
        "diagnostics_prefix",
        "[FullFisherDiag]",
    )
    if not isinstance(diagnostics_prefix, str) or len(diagnostics_prefix.strip()) == 0:
        raise ConfigError(
            "full_model_fisher.diagnostics_prefix 必须是非空字符串，"
            f"当前值：{diagnostics_prefix}"
        )


def _validate_fisher_only_global_config(
    cfg: Mapping[str, Any],
    non_expert_method: str,
) -> None:
    """
    检查 fisher_only_global 配置。

    fisher_only_global 本身第一版很简单：
        score_i = num_samples_i * fisher_strength_i

    主要强约束：
        1. 只能用于 agg.non_expert.method；
        2. 必须开启 full_model_fisher.enabled=true。
    """
    global_cfg = cfg.get("fisher_only_global", {})

    if not isinstance(global_cfg, Mapping):
        raise ConfigError("fisher_only_global 必须是一个 dict。")

    enabled = bool(global_cfg.get("enabled", False))

    if non_expert_method == "fisher_only_global" and not enabled:
        raise ConfigError(
            "agg.non_expert.method=fisher_only_global 时，建议显式设置 "
            "fisher_only_global.enabled=true。"
        )

    if "enabled" in global_cfg and not isinstance(global_cfg.get("enabled"), bool):
        raise ConfigError(
            "fisher_only_global.enabled 必须是 bool，"
            f"当前值：{global_cfg.get('enabled')}"
        )


def _validate_fisher_history_wolf_global_config(
    cfg: Mapping[str, Any],
    non_expert_method: str,
) -> None:
    """
    检查 fisher_history_wolf_global 配置。

    fisher_history_wolf_global 用于 pure-FL：
        1. 使用 full_model_fisher 的 fisher_strength；
        2. 每个 client_id 维护一个历史状态；
        3. 用 WoLF-IMQ 抑制异常 observation；
        4. 输出整模型 client-wise 权重。
    """
    wolf_cfg = cfg.get("fisher_history_wolf_global", {})

    if not isinstance(wolf_cfg, Mapping):
        raise ConfigError("fisher_history_wolf_global 必须是一个 dict。")

    enabled = bool(wolf_cfg.get("enabled", False))

    if non_expert_method == "fisher_history_wolf_global" and not enabled:
        raise ConfigError(
            "agg.non_expert.method=fisher_history_wolf_global 时，建议显式设置 "
            "fisher_history_wolf_global.enabled=true。"
        )

    if "enabled" in wolf_cfg and not isinstance(wolf_cfg.get("enabled"), bool):
        raise ConfigError(
            "fisher_history_wolf_global.enabled 必须是 bool，"
            f"当前值：{wolf_cfg.get('enabled')}"
        )

    init_P = float(wolf_cfg.get("init_P", 1.0))
    if init_P <= 0:
        raise ConfigError(
            f"fisher_history_wolf_global.init_P 必须大于 0，当前值：{init_P}"
        )

    process_noise_Q = float(wolf_cfg.get("process_noise_Q", 0.05))
    if process_noise_Q < 0:
        raise ConfigError(
            "fisher_history_wolf_global.process_noise_Q 不能小于 0，"
            f"当前值：{process_noise_Q}"
        )

    observation_R = float(wolf_cfg.get("observation_R", 1.0))
    if observation_R <= 0:
        raise ConfigError(
            "fisher_history_wolf_global.observation_R 必须大于 0，"
            f"当前值：{observation_R}"
        )

    robust_c = float(wolf_cfg.get("robust_c", 2.0))
    if robust_c <= 0:
        raise ConfigError(
            f"fisher_history_wolf_global.robust_c 必须大于 0，当前值：{robust_c}"
        )

    eps = float(wolf_cfg.get("eps", 1.0e-8))
    if eps <= 0:
        raise ConfigError(
            f"fisher_history_wolf_global.eps 必须大于 0，当前值：{eps}"
        )

    _validate_fisher_history_wolf_global_diagnostics_config(wolf_cfg)


def _validate_fisher_history_wolf_global_diagnostics_config(
    wolf_cfg: Mapping[str, Any],
) -> None:
    """
    检查 Fisher-History-WoLF global 诊断配置。

    这些字段主要服务于：
        1. aggregation/fisher_history_wolf_global.py 生成 diagnostics
        2. fl/server.py 打印 [FullFisherWoLFDiag] 日志
    """
    bool_fields = [
        "diagnostics_enabled",
        "diagnostics_print",
        "diagnostics_include_records",
    ]

    for key in bool_fields:
        value = wolf_cfg.get(key)
        if not isinstance(value, bool):
            raise ConfigError(
                f"fisher_history_wolf_global.{key} 必须是 bool，当前值：{value}"
            )

    diagnostics_print_every = wolf_cfg.get(
        "diagnostics_print_every",
        1,
    )
    if (
        not isinstance(diagnostics_print_every, int)
        or isinstance(diagnostics_print_every, bool)
        or diagnostics_print_every <= 0
    ):
        raise ConfigError(
            "fisher_history_wolf_global.diagnostics_print_every 必须是正整数，"
            f"当前值：{diagnostics_print_every}"
        )

    diagnostics_prefix = wolf_cfg.get(
        "diagnostics_prefix",
        "[FullFisherWoLFDiag]",
    )
    if not isinstance(diagnostics_prefix, str) or len(diagnostics_prefix.strip()) == 0:
        raise ConfigError(
            "fisher_history_wolf_global.diagnostics_prefix 必须是非空字符串，"
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
