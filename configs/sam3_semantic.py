# 提供训练参数

# 继承基础配置文件。
# 后面的同名配置会覆盖或补充这些 base 配置。
_base_ = [
    "./_base_/runtime.py",
    "./_base_/optimizer.py",
    "./_base_/schedule.py",
    "./_base_/visualization.py",
    "./datasets/isaid.py",

# 被model_builder.py用来构建model和criterion
# 被build_training_components()调用
model = dict(
    # 定义模式: 当前配置只启用语义分割任务路径。
    task_mode="semantic",
    # SAM3 文本编码器使用的 BPE 词表路径, BPE就是分词用的词表文件
    bpe_path="assets/bpe_simple_vocab_16e6.txt.gz",
    # 本地 SAM3 预训练权重路径；load_from_hf=False 时使用该文件。
    checkpoint_path="weights/sam3.pt",
    # 是否从 Hugging Face 下载 SAM3 权重。
    load_from_hf=False,
    # 模型构建和训练默认使用的设备, 表示模型放在gpu上
    device="cuda",
    # False 表示构建后进入 train() 模式；True 则进入 eval() 模式。
    eval_mode=False,
    # 是否启用 torch.compile 编译模型子模块
    # torch.compile 是 PyTorch 的运行时优化功能，可能让模型跑得更快，但也可能带来调试困难或兼容问题
    compile=False,
    # 类别数较多时，将 prompt 按 chunk 分批处理以降低显存占用。
    prompt_chunk_size=8,

    # OpenCLIP 配置：提供遥感图像和类别文本的 CLIP 特征
    openclip_cfg=dict(
        # 启用 OpenCLIP 分支；当前 final mixer 依赖该分支。
        enabled=True,
        # 使用的 OpenCLIP 模型结构名称。--> 图像编码器结构 vit - large(大) -14(patch_size)
        model_name="ViT-L-14",
        # RemoteCLIP 预训练权重路径。
        pretrained="weights/RemoteCLIP-ViT-L-14.pt",
        # 普通clip输出整张图向量,这里指输出特征图, 因为要知道每一个位置的信息
        default_output="feat_map",

        # 使用 MaskCLIP 风格提取密集特征
        image_encoder_mode="maskclip",
        # 跳过最后1层，以保留更适合密集预测的视觉特征。
        # clip最后层偏向全局语义, 对密集分割不一定合适
        maskclip_skip_last_layers=1,

        # 类别名会填入这些模板中，形成 CLIP 文本 prompt。
        prompt_templates=[
            "a remote sensing image of {}.",
            "an aerial image of {}.",
            "a satellite image of {}.",
            "an overhead view of {}.",
        ],
        # 实际使用的 prompt template 数量。
        num_prompt_templates=4,
        # 将类别标签规范化后再送入 CLIP 文本编码器。
        normalize_label_for_clip=True,
    ),

    # final_mixer 负责融合 SAM3 语义输出、SAM3 像素特征和 CLIP 特征。
    final_mixer_cfg=dict(
        # 当前语义训练路径要求 final mixer 必须开启。
        enabled=True,

        # 每个类别维护的 class token 数量。即每个类别用32个向量表示
        num_class_tokens=32,
        # mask embedding 融合层数。
        fusion_layers=4,
        # attention head 数量。
        num_heads=8,
        # final mixer 内部 dropout 概率。
        dropout=0.1,
        # 是否预测类别 presence，用于判断某类别是否出现在图像中。
        presence_enabled=True,

        # CLIP-SAM 特征构建配置；当前新结构中主要用于配置兼容。
        clip_sam_feature_cfg=dict(
            enabled=True,
            # 不额外加入clip图像残差
            use_image_residual=False,
        ),

        # 将低分辨率 CLIP-SAM 对齐特征上采样到 SAM3 mask 分辨率。
        clip_sam_upsample_cfg=dict(
            enabled=True,
            window_size=8,
            shift_size=4,
            dropout=0.1,
        ),

        # class_code 的来源：对每个类别的多个 class token 求平均。
        class_code_cfg=dict(
            source="mean_class_tokens",
        ),

        # semantic prior(粗略分割结果) 用 presence logit 和 softmax mask 构造先验嵌入。
        semantic_prior_cfg=dict(
            type="presence_signed_softmax",
            # 控制 mask logits 缩放强度的温度参数。
            tau=16.0,
        ),

        # final mixer 中窗口注意力的窗口大小、位移和 dropout。
        window_attention_cfg=dict(
            window_size=8,
            shift_size=4,
            dropout=0.1,
        ),

        # mask head 通过 mask embedding 与 class_code 点积生成类别 mask。
        # 结果是: 这个像素属于这个类别的分数
        mask_head_cfg=dict(
            type="mask_embed_dot_class_code",
            direct_dot=True,
            # class token 关注空间特征前，对特征图做池化的步长, 用于下采样, 减小计算量。
            class_feature_pool_stride=4,
        ),
    ),

    # 冻结策略：只训练指定模块，其余模型参数冻结。
    freeze_cfg=dict(
        # True 表示先冻结全模型，再打开 trainable_modules。
        train_adapters_only=True,
        # 只有 final mixer 参与训练。
        trainable_modules=[
            "core.final_mixer",
        ],
        # train_adapters_only=True 时，该列表通常为空
        # 在这个模式下不需要列出来,因为代码已经冻结了整个模型
        frozen_modules=[],
    ),

    # 输出适配器配置；空字典表示使用默认 SemanticSegAdapter 参数。
    # adapter 的作用是把模型原始输出转换成训练/验证需要的格式
    adapter_cfg=dict(),

    # 语义分割损失函数配置。
    criterion_cfg=dict(
        # 标签中值为 255 的像素不参与损失和指标计算。
        ignore_index=255,

        # final 输出的 BCE、Dice、CE 损失权重, 是loss算出来之后再加到总loss里的权重。
        final_bce_weight=0.4,
        final_dice_weight=1.0,
        final_ce_weight=0.4,
        # ignore 区域对应 BCE 项的权重。
        final_ignore_bce_weight=0.0,

        # presence 分类损失权重
        presence_loss_weight=1.0,
        # final mixer 有多层，每一层都可能输出 presence, 这些是中间层 presence 辅助损失权重
        presence_layer_loss_weights=[0.02, 0.05, 0.1, 0.2],

        # 中间 mask 层辅助损失的总体权重和逐层权重。
        mask_layer_loss_weight=1.0,
        mask_layer_weights=[0.1, 0.2, 0.4],

        # BCE 类别均衡权重的裁剪范围，避免极端类别频率导致权重过大或过小。
        bce_class_balance_clamp_min=0.2,
        bce_class_balance_clamp_max=5.0,

        # CE 类别均衡权重的裁剪范围。
        ce_class_balance_clamp_min=0.2,
        ce_class_balance_clamp_max=5.0,

        # dice loss主要按类别/样本 pair做平均, 没有根据像素数量给没一个类别乘动态权重, 
        # 所以不需要类别均衡权重的裁剪范围
        
        # 数值稳定项，避免除零或 log(0)。
        eps=1e-6,
    ),
)

# 训练 dataloader 配置；数据集细节来自 _base_ 中的 datasets/isaid.py。
train_dataloader = dict(
    # 每个训练 batch 的图像数量。
    batch_size=2,
    # DataLoader worker 进程数。
    num_workers=8,
)

# 验证 dataloader 配置。
val_dataloader = dict(
    # 验证时通常使用较小 batch，降低显存占用并简化指标统计。
    batch_size=1,
    num_workers=8,
)

# 验证指标配置。
eval_cfg = dict(
    # 与训练损失保持一致，忽略标签值为 255 的像素。
    ignore_index=255,
    # 概率阈值；0.0 表示不额外过滤预测概率。
    prob_thd=0.0,
    # 背景类别索引。
    bg_idx=0,
    # 使用 score map 计算语义指标。
    use_score_map=True,
)

# 优化器配置，会被 engine/optimizer_builder.py 读取并实例化。
optim_wrapper = dict(
    optimizer=dict(
        # AdamW 解耦 weight decay，常用于 transformer/ViT 微调。
        type="AdamW",
        # 基础学习率。
        lr=1e-4,
        # 权重衰减系数。
        weight_decay=0.01,
        # Adam 系列优化器的一阶、二阶动量参数。
        betas=(0.9, 0.999),
        # 按参数名设置不同学习率或 weight decay。
        paramwise_cfg=dict(
            # norm 层不使用 weight decay。
            norm_decay_mult=0.0,
            custom_keys={
                # final mixer 是主要训练模块，因此使用更高学习率。
                "core.final_mixer": dict(
                    # 学习率乘4
                    lr_mult=4.0,
                    # 权重衰减乘4
                    decay_mult=1.0,
                ),
            },
        ),
    )
)

# 学习率调度器列表：先 warmup，再 cosine 衰减。
param_scheduler = [
    dict(
        # 线性 warmup，训练初期从较小学习率逐步升高。
        type="LinearLR",
        # 前1000 iter做warmup, 学习率从基础学习率的0.1慢慢升到正常值
        start_factor=0.1,
        total_iters=1000,
        end=0,
    ),
    dict(
        # 余弦退火调度，将学习率逐渐降到 eta_min。
        # 后 19000 iter 做余弦退火, 学习率慢慢降到 1e-6
        type="CosineAnnealingLR",
        T_max=19000,
        eta_min=1e-6,
    )
]

# Trainer 运行配置，由 model_builder.build_trainer_config_from_cfg 转成 TrainerConfig。
train_cfg = dict(
    # 最大训练迭代次数。
    max_iters=20000,
    # checkpoint 保存间隔。
    save_interval=1000,
    # 验证间隔；这里等于 max_iters，表示主要在训练结束时验证。
    eval_interval=20000,
    # 日志平滑窗口大小, 即日志统计最近20次的平均值。
    log_window_size=20,
    # 启用自动混合精度训练。
    # 训练时不全都用 float32，而是让一部分计算用更低精度，比如 float16 / bfloat16
    use_amp=True,
    # 梯度裁剪阈值，防止梯度爆炸。
    # 如果所有参数的梯度整体范数超过 0.1，就把它按比例缩小到 0.1 以内。
    grad_clip_norm=0.1,
    # 用于选择 best checkpoint 的监控指标。
    monitor="semantic.miou",
    # 指标越大越好。
    monitor_mode="max",
    # 最多保留的 checkpoint 数量。
    max_keep_ckpts=20,
    # 是否自动从工作目录中最近的 checkpoint 恢复。
    auto_resume=False,
    # Trainer 使用的设备。
    device="cuda",
)

# 测试时增强(tta)配置；enabled=False 表示默认不启用 TTA。
tta_cfg = dict(
    enabled=False,
    # 多尺度推理比例。
    scales=[0.75, 1.0, 1.25],
    # 翻转模式：无翻转、水平翻转、垂直翻转。
    flip_modes=["none", "h", "v"],
    # 输入尺寸会对齐到该倍数，适配 ViT patch/window 约束。
    size_divisor=14,
)
