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
    "fisher_kfac_expert",
    "history_wolf_kfac_score",
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

    # K-FAC / FedFisher expert 聚合配置
    cfg.setdefault("kfac", {})
    cfg["kfac"].setdefault("collect", False)
    cfg["kfac"].setdefault("weight_mode", "sample_weighted")
    cfg["kfac"].setdefault("solve_scope", "per_layer")
    cfg["kfac"].setdefault("solve_mode", "cg")
    cfg["kfac"].setdefault("server_steps", 5)
    cfg["kfac"].setdefault("server_lr", 0.01)
    cfg["kfac"].setdefault("adam_beta1", 0.9)
    cfg["kfac"].setdefault("adam_beta2", 0.99)
    cfg["kfac"].setdefault("adam_eps", 0.01)
    cfg["kfac"].setdefault("cg_tol", 1.0e-8)
    cfg["kfac"].setdefault("damping", 0.0)
    cfg["kfac"].setdefault("use_damping", False)
    cfg["kfac"].setdefault("min_count", 1)
    cfg["kfac"].setdefault("fallback", "none")
    cfg["kfac"].setdefault("include_bias", True)
    cfg["kfac"].setdefault("fisher_timing", "after_train")
    cfg["kfac"].setdefault("model_mode", "eval")

    # 是否在 K-FAC/Fisher 采集时临时关闭训练集随机增强。
    # 默认 False，避免影响已有 fisher_kfac_expert 实验；
    # 新的 history_wolf_kfac_score 实验配置里可以显式设为 true。
    cfg["kfac"].setdefault("disable_augmentation_for_collect", False)

    cfg["kfac"].setdefault("max_batches", 0)
    cfg["kfac"].setdefault("expert_name_pattern", "experts.")
    cfg["kfac"].setdefault("use_server_validation", False)
    cfg["kfac"].setdefault("model_selection", "final_step")
    cfg["kfac"].setdefault("log_detail", True)

    # History-WoLF K-FAC Score 专家聚合配置
    # 按照项目的极致解耦风格，单独放在顶层配置块里，
    # 不塞到 agg.expert 下面。
    cfg.setdefault("history_wolf_kfac_score", {})
    cfg["history_wolf_kfac_score"].setdefault("fisher_score_enabled", True)
    cfg["history_wolf_kfac_score"].setdefault("history_filter_enabled", True)
    cfg["history_wolf_kfac_score"].setdefault("min_active_count", 1)
    cfg["history_wolf_kfac_score"].setdefault("min_valid_clients", 2)
    cfg["history_wolf_kfac_score"].setdefault("fallback", "keep_global")
    cfg["history_wolf_kfac_score"].setdefault("active_count_ref", 32)
    cfg["history_wolf_kfac_score"].setdefault("rho", 0.95)
    cfg["history_wolf_kfac_score"].setdefault("c_wolf", 2.5)
    cfg["history_wolf_kfac_score"].setdefault("min_obs_scale", 0.05)
    cfg["history_wolf_kfac_score"].setdefault("seen_ref", 5)
    cfg["history_wolf_kfac_score"].setdefault("q_scale", 0.05)
    cfg["history_wolf_kfac_score"].setdefault("tau_cur", 1.0)
    cfg["history_wolf_kfac_score"].setdefault("tau_hist", 1.0)
    cfg["history_wolf_kfac_score"].setdefault("init_P", 1.0)
    cfg["history_wolf_kfac_score"].setdefault("eps", 1.0e-8)

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

    # 控制台是否显示 tqdm 进度条。
    # 进度条只用于人看，不应该写入 train.log。
    cfg["logging"].setdefault("progress_bar", True)

    # 默认只在交互式终端显示进度条。
    # 如果用 nohup / 重定向跑实验，建议保持 False，避免输出混乱。
    cfg["logging"].setdefault("progress_in_non_tty", False)

    # 控制台每轮短摘要。
    # 例如：[Round 001] train_loss=... | test_acc=...
    cfg["logging"].setdefault("console_round_summary", True)

    # train.log 每轮详细摘要。
    # 例如：RoundMetrics / Clients / Agg / Client。
    cfg["logging"].setdefault("file_round_detail", True)

    # 是否记录本轮选择了哪些客户端。
    # 输出示例：
    # [Clients] round=1 ids=[0,4,9,6,7,3,2,8,1,5]
    cfg["logging"].setdefault("log_round_clients", True)

    # 是否在 train.log 中打印每个客户端一行诊断信息。
    # 输出内容包括样本数、训练指标、聚合权重、expert usage。
    cfg["logging"].setdefault("log_client_table", True)

    # 是否记录客户端训练指标。
    # 当前 server.py 的客户端表会默认包含 train_loss / train_acc。
    # 这个开关先保留，后续如果想拆分更细日志可以继续用。
    cfg["logging"].setdefault("log_client_metrics", True)

    # 是否在 train.log 记录每个客户端的聚合权重。
    # 对诊断 expert 聚合很重要，建议默认打开。
    cfg["logging"].setdefault("log_agg_weights", True)

    # 如果所有客户端权重近似相等，压缩显示为：
    # weights=uniform(each=0.1000)
    # 避免日志里出现一长串 0.10000000000000002。
    cfg["logging"].setdefault("compact_uniform_weights", True)

    # 是否在客户端本地训练结束后，额外统计 expert 使用情况。
    # 统计结果会进入 ClientUpdate.extra["expert_usage"]。
    cfg["logging"].setdefault("collect_expert_usage", True)

    # expert usage 最多统计多少个 batch。
    # 0 表示使用完整客户端 train_loader 统计，更准但更慢。
    # 如果觉得慢，可以在实验配置里改成 5 或 10。
    cfg["logging"].setdefault("expert_usage_max_batches", 0)

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


def _make_unique_run_name(run_name: str, output_dir: Path) -> str:
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

    这个函数只检查通用配置。
    后面新增 Fisher / history / Bayes 时，可以继续拆出新的 validate 函数。
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

    _validate_kfac_config(cfg)
    _validate_history_wolf_kfac_score_config(cfg)


def _validate_kfac_config(cfg: Mapping[str, Any]) -> None:
    """检查 K-FAC / FedFisher expert 聚合配置。"""
    kfac_cfg = cfg.get("kfac", {})

    if not isinstance(kfac_cfg, Mapping):
        raise ConfigError("kfac 必须是 dict。")

    weight_mode = str(kfac_cfg.get("weight_mode", "sample_weighted")).lower().strip()
    if weight_mode not in {"routed_count", "sample_weighted", "uniform"}:
        raise ConfigError(
            f"不支持的 kfac.weight_mode：{weight_mode}。"
            "当前支持：routed_count, sample_weighted, uniform"
        )

    solve_scope = str(kfac_cfg.get("solve_scope", "per_layer")).lower().strip()
    if solve_scope not in {"per_layer", "global_expert"}:
        raise ConfigError(
            f"不支持的 kfac.solve_scope：{solve_scope}。"
            "当前支持：per_layer, global_expert"
        )

    solve_mode = str(kfac_cfg.get("solve_mode", "cg")).lower().strip()
    if solve_mode not in {"cg", "gd", "adam"}:
        raise ConfigError(
            f"不支持的 kfac.solve_mode：{solve_mode}。"
            "当前支持：cg, gd, adam"
        )

    if solve_scope == "global_expert" and solve_mode == "cg":
        raise ConfigError(
            "kfac.solve_scope=global_expert 时不建议使用 solve_mode=cg。"
            "请使用 gd 或 adam。"
        )

    if solve_scope == "per_layer" and solve_mode in {"gd", "adam"}:
        raise ConfigError(
            "kfac.solve_scope=per_layer 当前只支持 solve_mode=cg。"
            "如果要使用 gd/adam，请设置 solve_scope=global_expert。"
        )

    server_steps = int(kfac_cfg.get("server_steps", 5))
    if server_steps < 0:
        raise ConfigError(
            f"kfac.server_steps 不能小于 0，当前值：{server_steps}"
        )

    server_lr = float(kfac_cfg.get("server_lr", 0.01))
    if server_lr <= 0:
        raise ConfigError(
            f"kfac.server_lr 必须大于 0，当前值：{server_lr}"
        )

    adam_beta1 = float(kfac_cfg.get("adam_beta1", 0.9))
    adam_beta2 = float(kfac_cfg.get("adam_beta2", 0.99))

    if not (0.0 <= adam_beta1 < 1.0):
        raise ConfigError(
            f"kfac.adam_beta1 必须在 [0, 1) 范围内，当前值：{adam_beta1}"
        )

    if not (0.0 <= adam_beta2 < 1.0):
        raise ConfigError(
            f"kfac.adam_beta2 必须在 [0, 1) 范围内，当前值：{adam_beta2}"
        )

    adam_eps = float(kfac_cfg.get("adam_eps", 0.01))
    if adam_eps <= 0:
        raise ConfigError(
            f"kfac.adam_eps 必须大于 0，当前值：{adam_eps}"
        )

    cg_tol = float(kfac_cfg.get("cg_tol", 1.0e-8))
    if cg_tol < 0:
        raise ConfigError(
            f"kfac.cg_tol 不能小于 0，当前值：{cg_tol}"
        )

    damping = float(kfac_cfg.get("damping", 0.0))
    if damping < 0:
        raise ConfigError(
            f"kfac.damping 不能小于 0，当前值：{damping}"
        )

    min_count = int(kfac_cfg.get("min_count", 1))
    if min_count <= 0:
        raise ConfigError(
            f"kfac.min_count 必须大于 0，当前值：{min_count}"
        )

    max_batches = int(kfac_cfg.get("max_batches", 0))
    if max_batches < 0:
        raise ConfigError(
            f"kfac.max_batches 不能小于 0，当前值：{max_batches}"
        )

    fallback = str(kfac_cfg.get("fallback", "none")).lower().strip()
    if fallback not in {"none", "sample_weighted"}:
        raise ConfigError(
            f"不支持的 kfac.fallback：{fallback}。"
            "当前支持：none, sample_weighted"
        )

    fisher_timing = str(kfac_cfg.get("fisher_timing", "after_train")).lower().strip()
    if fisher_timing != "after_train":
        raise ConfigError(
            f"当前只支持 kfac.fisher_timing=after_train，当前值：{fisher_timing}"
        )

    model_mode = str(kfac_cfg.get("model_mode", "eval")).lower().strip()
    if model_mode not in {"eval", "train"}:
        raise ConfigError(
            f"不支持的 kfac.model_mode：{model_mode}。"
            "当前支持：eval, train"
        )

    model_selection = str(kfac_cfg.get("model_selection", "final_step")).lower().strip()
    if model_selection != "final_step":
        raise ConfigError(
            "当前主实验不支持 server validation 选 best，"
            f"kfac.model_selection 必须是 final_step，当前值：{model_selection}"
        )

    use_server_validation = bool(kfac_cfg.get("use_server_validation", False))
    if use_server_validation:
        raise ConfigError(
            "当前主实验不使用 server validation，"
            "请设置 kfac.use_server_validation=false。"
        )


def _validate_history_wolf_kfac_score_config(cfg: Mapping[str, Any]) -> None:
    """检查 History-WoLF K-FAC Score 专家聚合配置。"""
    history_cfg = cfg.get("history_wolf_kfac_score", {})

    if not isinstance(history_cfg, Mapping):
        raise ConfigError("history_wolf_kfac_score 必须是 dict。")

    min_active_count = int(history_cfg.get("min_active_count", 1))
    if min_active_count < 0:
        raise ConfigError(
            "history_wolf_kfac_score.min_active_count 不能小于 0，"
            f"当前值：{min_active_count}"
        )

    min_valid_clients = int(history_cfg.get("min_valid_clients", 2))
    if min_valid_clients <= 0:
        raise ConfigError(
            "history_wolf_kfac_score.min_valid_clients 必须大于 0，"
            f"当前值：{min_valid_clients}"
        )

    fallback = str(history_cfg.get("fallback", "keep_global")).lower().strip()
    if fallback not in {"keep_global", "uniform"}:
        raise ConfigError(
            f"不支持的 history_wolf_kfac_score.fallback：{fallback}。"
            "当前支持：keep_global, uniform"
        )

    active_count_ref = float(history_cfg.get("active_count_ref", 32))
    if active_count_ref <= 0:
        raise ConfigError(
            "history_wolf_kfac_score.active_count_ref 必须大于 0，"
            f"当前值：{active_count_ref}"
        )

    rho = float(history_cfg.get("rho", 0.95))
    if not (0.0 <= rho <= 1.0):
        raise ConfigError(
            "history_wolf_kfac_score.rho 必须在 [0, 1] 范围内，"
            f"当前值：{rho}"
        )

    c_wolf = float(history_cfg.get("c_wolf", 2.5))
    if c_wolf <= 0:
        raise ConfigError(
            "history_wolf_kfac_score.c_wolf 必须大于 0，"
            f"当前值：{c_wolf}"
        )

    min_obs_scale = float(history_cfg.get("min_obs_scale", 0.05))
    if min_obs_scale <= 0:
        raise ConfigError(
            "history_wolf_kfac_score.min_obs_scale 必须大于 0，"
            f"当前值：{min_obs_scale}"
        )

    seen_ref = float(history_cfg.get("seen_ref", 5))
    if seen_ref <= 0:
        raise ConfigError(
            "history_wolf_kfac_score.seen_ref 必须大于 0，"
            f"当前值：{seen_ref}"
        )

    q_scale = float(history_cfg.get("q_scale", 0.05))
    if q_scale < 0:
        raise ConfigError(
            "history_wolf_kfac_score.q_scale 不能小于 0，"
            f"当前值：{q_scale}"
        )

    tau_cur = float(history_cfg.get("tau_cur", 1.0))
    if tau_cur <= 0:
        raise ConfigError(
            "history_wolf_kfac_score.tau_cur 必须大于 0，"
            f"当前值：{tau_cur}"
        )

    tau_hist = float(history_cfg.get("tau_hist", 1.0))
    if tau_hist <= 0:
        raise ConfigError(
            "history_wolf_kfac_score.tau_hist 必须大于 0，"
            f"当前值：{tau_hist}"
        )

    init_P = float(history_cfg.get("init_P", 1.0))
    if init_P <= 0:
        raise ConfigError(
            "history_wolf_kfac_score.init_P 必须大于 0，"
            f"当前值：{init_P}"
        )

    eps = float(history_cfg.get("eps", 1.0e-8))
    if eps <= 0:
        raise ConfigError(
            "history_wolf_kfac_score.eps 必须大于 0，"
            f"当前值：{eps}"
        )


def _require_positive_int(cfg: Mapping[str, Any], key: str) -> None:
    """检查某个字段是否为正整数。"""
    value = cfg.get(key)

    if not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{key} 必须是正整数，当前值：{value}")


def _require_non_negative_int(cfg: Mapping[str, Any], key: str) -> None:
    """检查某个字段是否为非负整数。"""
    value = cfg.get(key)

    if not isinstance(value, int) or value < 0:
        raise ConfigError(f"{key} 必须是非负整数，当前值：{value}")