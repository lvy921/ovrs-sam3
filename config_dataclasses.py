from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


# 冻结策略配置，由 model_builder.apply_freeze_cfg 读取。
@dataclass
class FreezeConfig:
    # True 表示先冻结全模型，再只打开 trainable_modules 中列出的模块。
    train_adapters_only: bool = False
    # train_adapters_only=True 时，允许参与训练的模块或参数名。
    trainable_modules: list[str] = field(default_factory=list)
    # train_adapters_only=False 时，显式冻结的模块或参数名。
    frozen_modules: list[str] = field(default_factory=list)


# OpenCLIP 分支配置，用于构建 RemoteCLIP/OpenCLIP 文本和图像编码器。
@dataclass
class OpenCLIPConfig:
    # 是否启用 OpenCLIP 分支；当前 final mixer 依赖 CLIP 图文特征。
    enabled: bool = False
    # OpenCLIP 模型结构名称。
    model_name: str = "ViT-L-14"
    # 本地 OpenCLIP/RemoteCLIP 预训练权重路径。
    pretrained: Optional[str] = None
    # 图像编码器默认输出类型，feat_map 表示输出空间特征图。
    default_output: str = "feat_map"

    # 图像编码器工作模式，maskclip 用于密集预测特征。
    image_encoder_mode: str = "maskclip"
    # MaskCLIP 模式下跳过最后若干层，以保留更适合分割的特征。
    maskclip_skip_last_layers: int = 1

    # 类别名会填入模板中的 {}，生成 CLIP 文本 prompt。
    prompt_templates: list[str] = field(
        default_factory=lambda: [
            "a remote sensing image of {}.",
            "an aerial image of {}.",
        ]
    )
    # 实际使用的 prompt template 数量。
    num_prompt_templates: int = 2

    # 是否在送入 CLIP 文本编码器前规范化类别标签。
    normalize_label_for_clip: bool = True


# CLIP-SAM 对齐特征构建配置；当前版本主要保留为配置结构的一部分。
@dataclass
class ClipSamFeatureConfig:
    enabled: bool = True
    # 是否保留 CLIP 图像残差分支。
    use_image_residual: bool = False


# CLIP-SAM 特征上采样配置，用于从 CLIP 低分辨率网格恢复到 mask 分辨率。
@dataclass
class ClipSamUpsampleConfig:
    enabled: bool = True
    # 窗口注意力的窗口大小。
    window_size: int = 8
    # shifted-window attention 的位移大小。
    shift_size: int = 4
    # 上采样注意力模块中的 dropout 概率。
    dropout: float = 0.1


# 类别代码 class_code 的构造方式配置。
@dataclass
class ClassCodeConfig:
    # mean_class_tokens 表示对每个类别的 class tokens 求平均得到 class_code。
    source: str = "mean_class_tokens"


# 语义先验配置，用于 final mixer 中构造 mask embedding 的先验项。
@dataclass
class SemanticPriorConfig:
    # presence_signed_softmax 使用 presence 信号修正 softmax mask 先验。
    type: str = "presence_signed_softmax"
    # mask logits 的温度/缩放参数。
    tau: float = 16.0


# final mixer 中空间窗口注意力的配置。
@dataclass
class WindowAttentionConfig:
    window_size: int = 8
    shift_size: int = 4
    dropout: float = 0.1


# final mixer 的 mask head 配置。
@dataclass
class MaskHeadConfig:
    # 通过 mask embedding 与 class_code 点积生成类别 logits。
    type: str = "mask_embed_dot_class_code"
    # True 表示直接使用点积形式，不额外引入复杂预测头。
    direct_dot: bool = True
    # class token 关注空间特征前，对特征图进行池化的步长。
    class_feature_pool_stride: int = 4


# final mixer 总配置，对应 models/final_mixer.py 中的 ClassTokenSemanticFinalMixer。
@dataclass
class FinalMixerConfig:
    # 当前语义训练路径要求 final mixer 开启。
    enabled: bool = True

    # 每个类别维护的 class token 数量。
    num_class_tokens: int = 32
    # mask embedding 融合层数。
    fusion_layers: int = 4
    # attention head 数量。
    num_heads: int = 8
    # final mixer 内部 dropout 概率。
    dropout: float = 0.1
    # 是否预测类别 presence，用于建模类别是否存在。
    presence_enabled: bool = True

    # 下列嵌套配置使用 default_factory，避免多个实例共享同一个可变对象。
    clip_sam_feature_cfg: ClipSamFeatureConfig = field(
        default_factory=ClipSamFeatureConfig
    )
    clip_sam_upsample_cfg: ClipSamUpsampleConfig = field(
        default_factory=ClipSamUpsampleConfig
    )
    class_code_cfg: ClassCodeConfig = field(default_factory=ClassCodeConfig)
    semantic_prior_cfg: SemanticPriorConfig = field(
        default_factory=SemanticPriorConfig
    )
    window_attention_cfg: WindowAttentionConfig = field(
        default_factory=WindowAttentionConfig
    )
    mask_head_cfg: MaskHeadConfig = field(default_factory=MaskHeadConfig)


# 语义分割损失配置，对应 losses/semantic_criterion.py。
@dataclass
class SemanticCriterionConfig:
    # 标签中该值对应的像素不参与损失和指标计算。
    ignore_index: int = 255

    # final 输出对应的 BCE、Dice、CE 损失权重。
    final_bce_weight: float = 0.4
    final_dice_weight: float = 1.0
    final_ce_weight: float = 0.4
    # ignore 区域 BCE 项的权重。
    final_ignore_bce_weight: float = 0.1

    # presence 分类损失权重。
    presence_loss_weight: float = 1.0
    # 各个中间 presence 层的辅助损失权重。
    presence_layer_loss_weights: Optional[list[float]] = field(
        default_factory=lambda: [0.02, 0.05, 0.1, 0.2]
    )

    # 中间 mask 层辅助损失总权重。
    mask_layer_loss_weight: float = 1.0
    # 各个中间 mask 层的辅助损失权重。
    mask_layer_weights: Optional[list[float]] = field(
        default_factory=lambda: [0.1, 0.2, 0.4]
    )

    # BCE 类别均衡权重的裁剪范围。
    bce_class_balance_clamp_min: float = 0.2
    bce_class_balance_clamp_max: float = 5.0

    # CE 类别均衡权重的裁剪范围。
    ce_class_balance_clamp_min: float = 0.2
    ce_class_balance_clamp_max: float = 5.0

    # 数值稳定项，避免除零或 log(0)。
    eps: float = 1e-6


# 输出 adapter 的配置占位；当前 SemanticSegAdapter 不需要额外参数。
@dataclass
class AdapterConfig:
    pass


# 模型构建总配置，由 SAM3ModelBuilder.build_config 标准化后使用。
@dataclass
class SegmentorBuildConfig:
    # 任务模式；当前实现主要支持 semantic。
    task_mode: str = "semantic"

    # SAM3 文本编码器 BPE 词表路径。
    bpe_path: Optional[str] = None
    # 本地 SAM3 checkpoint 路径。
    checkpoint_path: Optional[str] = None
    # checkpoint_path 为空时，是否从 Hugging Face 下载权重。
    load_from_hf: bool = True
    # 模型构建和训练默认设备。
    device: str = "cuda"
    # True 表示构建完成后进入 eval() 模式。
    eval_mode: bool = True
    # 是否对部分模块启用 torch.compile。
    compile: bool = False

    # 类别较多时按 chunk 处理 prompt，降低显存占用。
    prompt_chunk_size: Optional[int] = None

    # 嵌套配置：冻结策略、OpenCLIP、final mixer、loss 和 adapter。
    freeze_cfg: FreezeConfig = field(default_factory=FreezeConfig)
    openclip_cfg: OpenCLIPConfig = field(default_factory=OpenCLIPConfig)
    final_mixer_cfg: FinalMixerConfig = field(default_factory=FinalMixerConfig)
    criterion_cfg: SemanticCriterionConfig = field(
        default_factory=SemanticCriterionConfig
    )
    adapter_cfg: AdapterConfig = field(default_factory=AdapterConfig)


# Trainer 运行配置，由 model_builder.build_trainer_config_from_cfg 生成。
@dataclass
class TrainerConfig:
    # 最大训练迭代次数。
    max_iters: int = 10000
    # 日志统计的滑动窗口大小。
    log_window_size: int = 20
    # 是否启用 AMP 自动混合精度。
    use_amp: bool = True
    # 梯度裁剪阈值；None 表示不裁剪。
    grad_clip_norm: Optional[float] = 0.1

    # 输出目录与 checkpoint/验证间隔。
    save_dir: str = "./work_dirs/default"
    save_interval: int = 1000
    eval_interval: int = 1000

    # best checkpoint 的监控指标和比较方向。
    monitor: str = "semantic.miou"
    monitor_mode: str = "max"
    max_keep_ckpts: int = 5

    # Trainer 使用的设备，以及是否自动恢复最近 checkpoint。
    device: str = "cuda"
    auto_resume: bool = False

    # 测试时增强和验证器配置，来自顶层 cfg。
    tta_cfg: Optional[Dict] = None
    eval_cfg: Optional[Dict] = None


# checkpoint 管理器配置，对应 engine/checkpoint.py。
@dataclass
class CheckpointManagerConfig:
    # checkpoint 保存目录。
    save_dir: str
    # 用于选择 best checkpoint 的指标名。
    monitor: str = "total_loss"
    # min 表示越小越好，max 表示越大越好。
    mode: str = "min"
    # 最多保留的 checkpoint 数量。
    max_keep: int = 5
    # 是否维护 latest checkpoint。
    save_latest: bool = True
    # 是否维护 best checkpoint。
    save_best: bool = True


# 日志 hook 配置，对应 engine/hooks.py 中的 LoggerHook。
@dataclass
class LoggerHookConfig:
    # 训练日志打印间隔。
    interval: int = 20
    # 验证迭代日志打印间隔。
    val_interval: int = 50
    # 是否打印指标表格和逐类别指标。
    print_metric_tables: bool = True
    print_per_class_metrics: bool = True
    # hook 执行优先级。
    priority: int = 70


# 可视化配置，对应 engine/visualization.py。
@dataclass
class VisualizerConfig:
    # 是否启用可视化输出。
    enabled: bool = False
    # 可视化结果保存目录。
    save_dir: str = "./visualizations"
    # 可视化阶段，通常是 val。
    save_stage: str = "val"
    # 预测 mask 叠加到原图时的透明度。
    alpha: float = 0.45

    # 基础图像、预测和真值可视化开关。
    save_original: bool = True
    save_prediction: bool = True
    save_ground_truth: bool = True
    save_semantic_prediction: bool = True

    # 分数汇总和 score heatmap 输出开关。
    save_score_summary: bool = True
    save_score_heatmaps: bool = True
    # heatmap 使用的 matplotlib colormap 名称。
    heatmap_colormap: str = "turbo"

    # 是否保存 CLIP coarse 分割预测。
    save_clip_coarse_prediction: bool = True

    # 是否保存 SAM3 direct segmentation 结果及其阈值。
    save_sam3_direct_segmentation: bool = True
    sam3_direct_seg_threshold: float = 0.5

    # 是否保存 presence 分数和各层 presence 结果。
    save_presence_scores: bool = True
    save_presence_layers: bool = True

    # 是否保存 final mixer 各层 mask、heatmap、预测和 overlay。
    save_final_mixer_mask_layers: bool = True
    save_final_mixer_layer_heatmaps: bool = True
    save_final_mixer_layer_predictions: bool = True
    save_final_mixer_layer_overlays: bool = True
    # 限制每层 heatmap 可视化的类别数量；None 表示不限制。
    max_final_mixer_layer_heatmap_classes: Optional[int] = None

    # 可视化采样概率、每轮最多样本数和随机种子。
    vis_prob: float = 0.05
    max_samples_per_epoch: Optional[int] = 50
    vis_seed: int = 42

    # 每张图像的输出文件夹命名模板，以及可视化时忽略的标签值。
    image_folder_pattern: str = "image_{image_id:06d}"
    ignore_index: int = 255