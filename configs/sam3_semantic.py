_base_ = [
    "./_base_/runtime.py",
    "./_base_/optimizer.py",
    "./_base_/schedule.py",
    "./_base_/visualization.py",
    "./datasets/isaid.py",
]

model = dict(
    task_mode="semantic",
    bpe_path="assets/bpe_simple_vocab_16e6.txt.gz",
    checkpoint_path="weights/sam3.pt",
    load_from_hf=False,
    device="cuda",
    eval_mode=False,
    compile=False,
    prompt_chunk_size=8,

    openclip_cfg=dict(
        enabled=True,
        model_name="ViT-L-14",
        pretrained="weights/RemoteCLIP-ViT-L-14.pt",
        default_output="feat_map",

        image_encoder_mode="full_vit_dense",
        maskclip_skip_last_layers=1,

        extra_token_templates=[
            "a remote sensing image of {}.",
            "an aerial image of {}.",
            "a satellite image of {}.",
            "an overhead view of {}.",
        ],
        num_extra_tokens=4,
        normalize_label_for_clip=True,
    ),

    final_mixer_cfg=dict(
        enabled=True,
        hidden_dim=128,
        num_heads=8,
        dropout=0.1,
        gate_bias_init=3.0,

        class_attn_heads=4,
        class_attn_pooling_size=2,
    ),

    freeze_cfg=dict(
        train_adapters_only=True,
        trainable_modules=[
            "core.clip_native_image_to_text_attn",
            "core.clip_native_image_to_text_norm",

            "core.clip_hv_proj",

            "core.sam3_to_clip_hv_attn",
            "core.sam3_to_clip_hv_norm",

            "core.extra_type_embed",

            "core.extra_token_mask_query_proj",
            "core.extra_token_mask_memory_proj",
            "core.extra_token_logit_scale",

            "core.class_query_seed_proj",
            "core.class_query_encoder_cross_attn",
            "core.class_query_encoder_cross_attn_norm",
            "core.class_query_proj",

            "core.class_query_self_attn",
            "core.class_query_self_attn_norm",

            "core.score_fusion_block",

            "core.suppression_query_cross_attn",
            "core.suppression_query_cross_attn_norm",
            "core.suppression_logit_scale",
            "core.suppression_gate_bias",
        ],
        frozen_modules=[],
    ),

    adapter_cfg=dict(),

    criterion_cfg=dict(
        ignore_index=255,

        bce_weight=0.4,
        dice_weight=1.0,

        final_bce_weight=0.1,
        final_dice_weight=0.5,
        final_ce_weight=0.1,

        extra_token_aux_loss_weight=0.1,
        extra_token_aux_bce_weight=0.3,
        extra_token_aux_dice_weight=0.3,
        extra_token_aux_absent_weight=0.1,
        extra_token_aux_absent_topk_ratio=0.05,
        extra_token_aux_exclude_bg=False,
        extra_token_aux_bg_idx=0,

        suppression_absent_loss_weight=0.5,
        suppression_absent_topk_ratio=0.05,

        bce_class_balance_clamp_min=0.2,
        bce_class_balance_clamp_max=5.0,
        eps=1e-6,
    ),
)

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
)

eval_cfg = dict(
    ignore_index=255,
    prob_thd=0.0,
    bg_idx=0,
    use_score_map=True,
)

optim_wrapper = dict(
    optimizer=dict(
        type="AdamW",
        lr=3e-5,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        paramwise_cfg=dict(
            norm_decay_mult=0.0,
            custom_keys={
                "core.clip_native_image_to_text_attn": dict(lr_mult=2.0, decay_mult=1.0),
                "core.clip_native_image_to_text_norm": dict(lr_mult=2.0, decay_mult=0.0),

                "core.clip_hv_proj": dict(lr_mult=2.0, decay_mult=1.0),

                "core.sam3_to_clip_hv_attn": dict(lr_mult=2.0, decay_mult=1.0),
                "core.sam3_to_clip_hv_norm": dict(lr_mult=2.0, decay_mult=0.0),

                "core.extra_type_embed": dict(lr_mult=2.0, decay_mult=0.0),
                "core.extra_token_mask_query_proj": dict(lr_mult=3.0, decay_mult=1.0),
                "core.extra_token_mask_memory_proj": dict(lr_mult=3.0, decay_mult=1.0),
                "core.extra_token_logit_scale": dict(lr_mult=1.0, decay_mult=0.0),

                "core.class_query_seed_proj": dict(lr_mult=2.0, decay_mult=1.0),
                "core.class_query_encoder_cross_attn": dict(lr_mult=2.0, decay_mult=1.0),
                "core.class_query_encoder_cross_attn_norm": dict(lr_mult=2.0, decay_mult=0.0),
                "core.class_query_proj": dict(lr_mult=2.0, decay_mult=1.0),

                "core.class_query_self_attn": dict(lr_mult=2.0, decay_mult=1.0),
                "core.class_query_self_attn_norm": dict(lr_mult=2.0, decay_mult=0.0),

                "core.score_fusion_block": dict(lr_mult=2.0, decay_mult=1.0),

                "core.suppression_query_cross_attn": dict(lr_mult=2.0, decay_mult=1.0),
                "core.suppression_query_cross_attn_norm": dict(lr_mult=2.0, decay_mult=0.0),
                "core.suppression_logit_scale": dict(lr_mult=1.0, decay_mult=0.0),
                "core.suppression_gate_bias": dict(lr_mult=1.0, decay_mult=0.0),
            }
        ),
    )
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.1,
        total_iters=500,
        end=0,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=9500,
        eta_min=1e-6,
    )
]

train_cfg = dict(
    max_iters=10000,
    save_interval=1000,
    eval_interval=10000,
    log_window_size=20,
    use_amp=True,
    grad_clip_norm=0.1,
    monitor="semantic.miou",
    monitor_mode="max",
    max_keep_ckpts=10,
    auto_resume=False,
    device="cuda",
)

tta_cfg = dict(
    enabled=False,
    scales=[0.75, 1.0, 1.25],
    flip_modes=["none", "h", "v"],
    size_divisor=14,
)