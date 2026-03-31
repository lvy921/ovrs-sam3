# 这里只是示例，你可以把 classes 单独放到别的 config 里再 import

ld50k_classes = [
    # 在这里填你的 40 个类别名
    # 例如：
    # 'water',
    # 'building',
    # 'road',
    # ...
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True,
    dataset=dict(
        type='data.dataset.OVSemanticSegDataset',
        img_dir='data/ld50k/img_dir/train',
        ann_dir='data/ld50k/ann_dir/train',
        classes=ld50k_classes,
        ignore_index=255,
        reduce_zero_label=True,
        return_raw_image=True,
        transforms=[
            dict(type='ToTensor'),
            dict(type='ConvertImageDtype'),
            dict(type='ResizeLongestSide', long_side=1008),
        ],
    ),
    collate_fn=dict(
        type='data.collate.OVSemanticCollator',
        pad_size_divisor=14,
        label_pad_value=255,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    shuffle=False,
    pin_memory=True,
    persistent_workers=True,
    dataset=dict(
        type='data.dataset.OVSemanticSegDataset',
        img_dir='data/ld50k/img_dir/val',
        ann_dir='data/ld50k/ann_dir/val',
        classes=ld50k_classes,
        ignore_index=255,
        reduce_zero_label=True,
        return_raw_image=True,
        transforms=[
            dict(type='ToTensor'),
            dict(type='ConvertImageDtype'),
            dict(type='ResizeLongestSide', long_side=1008),
        ],
    ),
    collate_fn=dict(
        type='data.collate.OVSemanticCollator',
        pad_size_divisor=14,
        label_pad_value=255,
    ),
)