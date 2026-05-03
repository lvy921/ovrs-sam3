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

    freeze_cfg=dict(
        train_adapters_only=True,
        trainable_modules=[
            # CLIP native image -> CLIP text attention
            "core.clip_native_image_to_text_attn",
            "core.clip_native_image_to_text_norm",

            # CLIP native space -> SAM3 hidden dim
            "core.clip_hv_proj",

            # SAM3 text -> CLIP-enhanced image-token attention
            "core.sam3_to_clip_hv_attn",
            "core.sam3_to_clip_hv_norm",

            # distinguish extra tokens from original SAM3 text tokens
            "core.extra_type_embed",

            # extra token auxiliary mask head
            "core.extra_token_mask_query_proj",
            "core.extra_token_mask_memory_proj",
            "core.extra_token_logit_scale",

            # new presence seed
            "core.presence_seed_proj",

            # presence branch
            "core.presence_cross_attn",
            "core.presence_cross_attn_norm",
            "core.presence_head",

            # final logits modulation
            "adapter.presence_modulation_alpha",
        ],
        frozen_modules=[],
    ),

    adapter_cfg=dict(
        presence_base=0.1,
        init_presence_modulation_alpha=1.0,
    ),

    criterion_cfg=dict(
        ignore_index=255,

        # original semantic mask supervision
        bce_weight=0.4,
        dice_weight=1.0,

        # weakened presence supervision
        presence_bce_weight=0.2,
        presence_pos_weight=1.0,

        # final score map supervision
        final_bce_weight=0.1,
        final_dice_weight=0.1,
        final_ce_weight=0.1,

        # extra token auxiliary supervision
        extra_token_aux_loss_weight=0.1,
        extra_token_aux_bce_weight=0.3,
        extra_token_aux_dice_weight=0.3,
        extra_token_aux_absent_weight=0.1,
        extra_token_aux_absent_topk_ratio=0.05,
        extra_token_aux_exclude_bg=False,
        extra_token_aux_bg_idx=0,

        # other loss hyper-parameters
        bce_class_balance_clamp_min=0.2,
        bce_class_balance_clamp_max=5.0,
        eps=1e-6,
    ),
)

train_dataloader = dict(
    batch_size=4,
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
                # -------- CLIP native image -> text attention --------
                "core.clip_native_image_to_text_attn": dict(lr_mult=2.0, decay_mult=1.0),
                "core.clip_native_image_to_text_norm": dict(lr_mult=2.0, decay_mult=0.0),

                # -------- CLIP hidden visual tokens -> SAM3 hidden dim --------
                "core.clip_hv_proj": dict(lr_mult=2.0, decay_mult=1.0),

                # -------- SAM3 text token attends to CLIP-enhanced visual tokens --------
                "core.sam3_to_clip_hv_attn": dict(lr_mult=2.0, decay_mult=1.0),
                "core.sam3_to_clip_hv_norm": dict(lr_mult=2.0, decay_mult=0.0),

                # -------- extra token identity and auxiliary mask head --------
                "core.extra_type_embed": dict(lr_mult=2.0, decay_mult=0.0),
                "core.extra_token_mask_query_proj": dict(lr_mult=3.0, decay_mult=1.0),
                "core.extra_token_mask_memory_proj": dict(lr_mult=3.0, decay_mult=1.0),
                "core.extra_token_logit_scale": dict(lr_mult=1.0, decay_mult=0.0),

                # -------- presence branch, keep moderate --------
                "core.presence_seed_proj": dict(lr_mult=2.0, decay_mult=1.0),
                "core.presence_cross_attn": dict(lr_mult=2.0, decay_mult=1.0),
                "core.presence_cross_attn_norm": dict(lr_mult=2.0, decay_mult=0.0),
                "core.presence_head": dict(lr_mult=2.0, decay_mult=1.0),

                # -------- final logits modulation --------
                "adapter.presence_modulation_alpha": dict(lr_mult=1.0, decay_mult=0.0),
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