from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

# 当使用 `python tools/train.py ...` 直接运行本文件时，__package__
# 通常为 None 或空字符串；此时普通的相对导入 `from ..xxx` 会失败。
if __package__ in (None, ""):
    # 当前文件位于 repo_root/tools/train.py，因此 parents[1] 是仓库根目录。
    repo_root = Path(__file__).resolve().parents[1]
    from importlib import import_module
    import types

    # 构造一个仅在当前 Python 进程中存在的临时包名。
    # 仓库目录名可能包含连字符，不能直接作为合法 Python 包名使用。
    package_name = "_ovrs_sam3_localpkg"
    if package_name not in sys.modules:
        # 创建模块对象，并通过 __path__ 将其声明为“包”。
        # 后续导入 _ovrs_sam3_localpkg.data.build 时，
        # Python 会在 repo_root/data/build.py 中查找模块文件。
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(repo_root)]
        sys.modules[package_name] = pkg

    # 动态导入仓库内部模块，并取出 main() 后续需要使用的函数/类。
    build_dataloader = import_module(f"{package_name}.data.build").build_dataloader
    Config = import_module(f"{package_name}.engine.config").Config
    _opt_mod = import_module(f"{package_name}.engine.optimizer_builder")
    build_optimizer = _opt_mod.build_optimizer
    build_scheduler = _opt_mod.build_scheduler
    _trainer_mod = import_module(f"{package_name}.engine.trainer")
    Trainer = _trainer_mod.Trainer

    _builder_mod = import_module(f"{package_name}.model_builder")
    build_training_components = _builder_mod.build_training_components
    build_train_runtime_components = _builder_mod.build_train_runtime_components
else:
    # 当本文件作为包内模块运行或被导入时，使用标准相对导入。
    from ..data.build import build_dataloader
    from ..engine.config import Config
    from ..engine.optimizer_builder import build_optimizer, build_scheduler
    from ..engine.trainer import Trainer
    from ..model_builder import (
        build_train_runtime_components,
        build_training_components,
    )


def set_seed(seed: int = 42):
    # 固定 Python 标准库 random 的随机数状态。
    random.seed(seed)
    # 固定 NumPy 的随机数状态。
    np.random.seed(seed)
    # 固定 PyTorch CPU 上的随机数状态。
    torch.manual_seed(seed)
    # 固定所有 CUDA 设备上的随机数状态。
    torch.cuda.manual_seed_all(seed)


class _DotDict(dict):
    # 将 cfg.xxx 形式的属性访问转发为 cfg.get("xxx")。
    __getattr__ = dict.get
    # 将 cfg.xxx = value 转发为 cfg["xxx"] = value。
    __setattr__ = dict.__setitem__
    # 将 del cfg.xxx 转发为 del cfg["xxx"]。
    __delattr__ = dict.__delitem__


def _to_dotdict(obj: Any):
    # 递归转换 dict，使配置支持 cfg.model、cfg.train_cfg 等点号访问。
    if isinstance(obj, dict):
        return _DotDict({k: _to_dotdict(v) for k, v in obj.items()})
    # list 本身保持为 list，但内部元素继续递归转换。
    if isinstance(obj, list):
        return [_to_dotdict(x) for x in obj]
    # Tensor、数字、字符串、None 等非容器对象保持原值。
    return obj


def build_log_getters(cfg) -> List[object]:
    # 构造额外日志提取函数，Trainer 会在记录日志时调用这些 getter。
    def project_log_getter(trainer):
        out = {}

        model = trainer.model
        # 兼容 torch.nn.DataParallel / DistributedDataParallel。
        # 被包装时真实模型位于 model.module；未包装时保持 model 本身。
        model = getattr(model, "module", model)
        # 当前项目的训练模型通常是 SAM3Segmentor，核心网络挂在 core 上。
        core = getattr(model, "core", None)

        if core is None:
            return out

        # 兼容旧结构中的 CLIP-SAM 特征融合模块；不存在则不记录该项。
        feature_builder = getattr(
            core,
            "global_clip_sam_feature_builder",
            None,
        )
        if feature_builder is None:
            return out

        alpha = getattr(feature_builder, "alpha", None)
        if alpha is not None:
            # detach 避免日志记录把 alpha 接入计算图。
            alpha = alpha.detach()
            if alpha.numel() == 1:
                out["clip_sam_feature_alpha"] = float(alpha.item())
            else:
                out["clip_sam_feature_alpha_mean"] = float(alpha.float().mean().item())

        return out

    return [project_log_getter]


def _unwrap_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    # checkpoint 应当是字典；否则无法判断其中的模型参数结构。
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")

    # 常见保存格式一：{"model": state_dict, ...}
    if "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]

    # 常见保存格式二：{"state_dict": state_dict, ...}
    if "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]

    # 常见保存格式三：checkpoint 本身就是 state_dict。
    if all(isinstance(k, str) for k in obj.keys()):
        return obj

    raise ValueError("Cannot find a valid state_dict in the checkpoint.")


def _strip_prefix_if_present(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    # 空 state_dict 不需要处理。
    if not state_dict:
        return state_dict

    keys = list(state_dict.keys())
    # 多卡包装保存的权重名可能统一带有 "module." 前缀。
    # 只有当所有 key 都有该前缀时才统一移除，避免误删部分参数名。
    if all(k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in state_dict.items()}

    return state_dict


def load_model_weights_only(
    model: torch.nn.Module,
    path: str,
    strict: bool = False,
) -> Dict[str, Any]:
    # map_location="cpu" 先将权重加载到 CPU，避免加载阶段占用 GPU 显存。
    ckpt = torch.load(path, map_location="cpu")
    # 从多种 checkpoint 保存格式中提取真正的模型 state_dict。
    state_dict = _unwrap_state_dict(ckpt)
    # 兼容 DataParallel / DDP 保存出的 "module.xxx" 参数名。
    state_dict = _strip_prefix_if_present(state_dict, "module.")

    # strict=False 允许当前模型与 checkpoint 参数不完全一致，
    # 适用于迁移学习或只加载部分权重的场景。
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

    print(f"Loaded model weights from {path}")
    if len(missing_keys) > 0:
        print(f"Missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        print(f"Unexpected keys: {unexpected_keys}")

    return {
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }


def main():
    # 创建命令行参数解析器，用于解析训练配置路径和运行选项。
    parser = argparse.ArgumentParser(
        description="Train/Eval SAM3 semantic segmentor with iter-based training."
    )
    # 位置参数：配置文件路径，调用脚本时必须提供。
    parser.add_argument("config", type=str, help="path to config file")
    # 可选参数：覆盖配置文件中的输出目录。
    parser.add_argument("--work-dir", type=str, default=None)
    # 从完整 checkpoint 恢复训练状态，包含模型、优化器、scheduler、iter 等。
    parser.add_argument("--resume-from", type=str, default=None)
    # 仅加载模型权重，不恢复优化器状态和训练进度。
    parser.add_argument("--load-model-from", type=str, default=None)
    # 启用后由 checkpoint_manager 自动查找最近的 checkpoint 恢复。
    parser.add_argument("--auto-resume", action="store_true")
    # 覆盖配置文件中的随机种子。
    parser.add_argument("--seed", type=int, default=None)
    # 只执行验证流程，不进入训练循环。
    parser.add_argument("--eval-only", action="store_true", help="only run validation")
    parser.add_argument(
        "--eval-iter",
        type=int,
        default=0,
        # eval-only 且不 resume 时，用该值作为日志和可视化中的迭代编号。
        help="iter id used in eval-only outputs/logging",
    )
    args = parser.parse_args()

    # 两个参数语义冲突：resume 恢复完整训练状态，load-model 只加载权重。
    if args.resume_from is not None and args.load_model_from is not None:
        raise ValueError("--resume-from and --load-model-from cannot be used together.")

    # 读取配置文件，并将配置递归转换为支持点号访问的字典对象。
    cfg = Config.fromfile(args.config)
    cfg = _to_dotdict(cfg)

    # 命令行 --seed 优先级高于配置文件；都未设置时默认使用 42。
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)

    # 根据 cfg.model 构建训练模型和损失函数。
    model, criterion = build_training_components(**dict(cfg.model))

    # 仅加载模型权重，用于从预训练权重开始微调。
    if args.load_model_from is not None:
        load_model_weights_only(model=model, path=args.load_model_from, strict=False)

    # 构建训练运行时组件：输出目录、Trainer 配置、hook、可视化器和 checkpoint 管理器。
    (
        work_dir,
        trainer_cfg,
        hooks,
        visualizer,
        checkpoint_manager,
    ) = build_train_runtime_components(
        cfg,
        work_dir_override=args.work_dir,
        auto_resume=args.auto_resume,
    )

    # 确保训练输出目录存在，用于保存日志、可视化结果和 checkpoint。
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        # eval-only 必须配置验证集，否则 Trainer 无法执行验证。
        if cfg.get("val_dataloader") is None:
            raise ValueError("val_dataloader is None, cannot run eval-only mode.")

        print("Building val_dataloader (eval-only)...")
        # 根据验证 dataloader 配置构建验证数据加载器。
        val_loader = build_dataloader(cfg.val_dataloader)

        # 验证模式下不需要 optimizer、train_dataloader 和 lr_scheduler。
        trainer = Trainer(
            model=model,
            optimizer=None,
            criterion=criterion,
            train_dataloader=None,
            val_dataloader=val_loader,
            lr_scheduler=None,
            cfg=trainer_cfg,
            hooks=hooks,
            checkpoint_manager=checkpoint_manager,
            visualizer=visualizer,
        )

        if args.resume_from:
            # 从 checkpoint 恢复模型与相关训练状态后再验证。
            trainer.resume_from(args.resume_from)
        else:
            # 不恢复 checkpoint 时，手动设置验证日志使用的 iter 编号。
            trainer.global_iter = int(args.eval_iter)

        # 执行验证流程并结束 main，避免继续进入训练分支。
        trainer.val()
        return

    print("Building train_dataloader...")
    # 根据训练 dataloader 配置构建训练数据加载器。
    train_loader = build_dataloader(cfg.train_dataloader)

    print("Building val_dataloader...")
    # 验证 dataloader 是可选项；未配置时训练过程不做周期性验证。
    val_loader = build_dataloader(cfg.val_dataloader) if cfg.get("val_dataloader") else None

    # 根据模型和配置构建优化器与学习率调度器。
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # 创建完整训练器，Trainer 负责训练循环、验证循环、日志、hook 和 checkpoint。
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        lr_scheduler=scheduler,
        cfg=trainer_cfg,
        hooks=hooks,
        checkpoint_manager=checkpoint_manager,
        visualizer=visualizer,
    )

    # 注册项目级额外日志字段，例如可学习融合权重 alpha。
    for getter in build_log_getters(cfg):
        trainer.register_log_getter(getter)

    if args.resume_from:
        # 用户显式指定 checkpoint 时，从该路径恢复完整训练状态。
        trainer.resume_from(args.resume_from)

    # 进入正式训练循环。
    trainer.train()


if __name__ == "__main__":
    main()