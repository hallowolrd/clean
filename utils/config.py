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
    "resnet_moe",
}

SUPPORTED_AGG_METHODS = {
    "uniform",
    "sample_weighted",
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


def save_config(cfg: ConfigNode | Mapping[str, Any], output_path: str | Path) -> None:
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


def _load_yaml_with_include(config_path: Path, stack: Optional[list[Path]] = None) -> Dict[str, Any]:
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

    # 当前配置覆盖 include 进来的配置
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
    cfg.setdefault("model", "resnet_moe")
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

    # 对未知数据集先返回 10，真正合法性检查在 _validate_config 里做
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