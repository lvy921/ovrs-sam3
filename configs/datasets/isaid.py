# iSAID 语义类别列表；因为 reduce_zero_label=True，背景 0 会被转成 ignore。
isaid_classes = [
    'ship',
    'store tank',
    'baseball diamond',
    'tennis court',
    'basketball court',
    'ground track field',
    'bridge',
    'large vehicle',
    'small vehicle',
    'helicopter',
    'swimming pool',
    'roundabout',
    'soccer ball field',
    'plane',
    'harbor',
]

# 训练 dataloader：定义数据路径、数据增强、collate 方式和 DataLoader 参数。
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True,
    dataset=dict(
        # 使用开放词汇语义分割数据集，样本会返回 image、label_map 和 class_texts。
        type='data.dataset.OVSemanticSegDataset',
        img_dir='data/datasets/iSAID/img_dir/train',
        ann_dir='data/datasets/iSAID/ann_dir/train',
        classes=isaid_classes,
        img_suffix='.png',
        seg_suffix='_instance_color_RGB.png',
        ignore_index=255,
        reduce_zero_label=True,
        return_raw_image=True,
        transforms=[
            # PIL/ndarray -> Tensor，并将图像缩放到 [0,1] 浮点范围。
            dict(type='ToTensor'),
            dict(type='ConvertImageDtype', dtype='float32', scale=True),

            # 多尺度随机缩放，增强遥感目标尺度变化的鲁棒性。
            dict(
                type='RandomResizeByRatio',
                base_scale=(1008, 1008),
                ratio_range=(0.5, 2.0),
                keep_ratio=True,
            ),

            # 随机裁剪到 SAM3 训练尺寸，并限制单一类别占比过高的 crop。
            dict(
                type='RandomCrop',
                crop_size=(1008, 1008),
                cat_max_ratio=0.75,
                ignore_index=255,
                pad_if_needed=True,
                image_pad_value=0.0,
            ),

            # 遥感俯视图方向不固定，因此水平/垂直翻转和 90 度旋转都合理。
            dict(type='RandomHorizontalFlip', prob=0.5),
            dict(type='RandomVerticalFlip', prob=0.5),
            dict(type='RandomRotate90', prob=0.5),

            # 将 [0,1] 图像归一化到大致 [-1,1]。
            dict(
                type='Normalize',
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ],
    ),
    collate_fn=dict(
        # batch 内图像 pad 到 14 的倍数，适配 ViT-L/14 patch 网格。
        type='data.collate.OVSemanticCollator',
        pad_size_divisor=14,
        label_pad_value=255,
    ),
)

# 验证 dataloader：关闭随机增强，只做固定尺寸 resize 和归一化。
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    shuffle=False,
    pin_memory=True,
    persistent_workers=True,
    dataset=dict(
        # 验证集使用相同类别顺序，保证输出通道与训练一致。
        type='data.dataset.OVSemanticSegDataset',
        img_dir='data/datasets/iSAID/img_dir/val',
        ann_dir='data/datasets/iSAID/ann_dir/val',
        classes=isaid_classes,
        img_suffix='.png',
        seg_suffix='_instance_color_RGB.png',
        ignore_index=255,
        reduce_zero_label=True,
        return_raw_image=True,
        transforms=[
            dict(type='ToTensor'),
            dict(type='ConvertImageDtype', dtype='float32', scale=True),
            dict(type='Resize', size=(1008, 1008), keep_ratio=False),
            dict(
                type='Normalize',
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ],
    ),
    collate_fn=dict(
        type='data.collate.OVSemanticCollator',
        pad_size_divisor=14,
        label_pad_value=255,
    ),
)