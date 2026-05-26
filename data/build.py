from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

from torch.utils.data import DataLoader


ConfigDict = Dict[str, Any]


# 根据字符串路径导入对象，例如 "data.dataset.OVSemanticSegDataset"。
def get_obj_from_string(path: str):
    module_name, obj_name = path.rsplit('.', 1)

    try:
        module = importlib.import_module(module_name)
        return getattr(module, obj_name)
    except ModuleNotFoundError as e:
        original_error = e

    # 当作为包内模块导入时，尝试补上当前根包名前缀作为 fallback。
    root_pkg = __package__.split('.')[0]

    fallback_module_name = f'{root_pkg}.{module_name}'
    try:
        module = importlib.import_module(fallback_module_name)
        return getattr(module, obj_name)
    except ModuleNotFoundError:
        raise original_error


def instantiate(cfg: Any, **extra_kwargs):
    # 根据 cfg["type"] 实例化类/函数；非 dict 对象直接原样返回。
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        return cfg
    cfg = dict(cfg)
    obj_type = cfg.pop('type')
    cls = get_obj_from_string(obj_type) if isinstance(obj_type, str) else obj_type
    cfg.update(extra_kwargs)
    return cls(**cfg)


def build_dataset(cfg: ConfigDict):
    # 构建 Dataset 实例。
    return instantiate(cfg)


def build_collate_fn(cfg: Optional[ConfigDict]):
    # collate_fn 可为空；为空时 DataLoader 使用默认 collate。
    if cfg is None:
        return None
    return instantiate(cfg)


def build_dataloader(cfg: ConfigDict):
    # 从配置中拆出 dataset/collate_fn，再把剩余参数传给 torch DataLoader。
    cfg = dict(cfg)
    dataset_cfg = cfg.pop('dataset')
    collate_fn_cfg = cfg.pop('collate_fn', None)
    dataset = build_dataset(dataset_cfg)
    collate_fn = build_collate_fn(collate_fn_cfg)
    return DataLoader(dataset=dataset, collate_fn=collate_fn, **cfg)
