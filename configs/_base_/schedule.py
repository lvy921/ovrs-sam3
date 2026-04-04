param_scheduler = dict(
    type='CosineAnnealingLR',
    T_max=12,
    eta_min=1e-6,
)

train_cfg = dict(
    max_epochs=12,
    log_window_size=20,
    use_amp=True,
    grad_clip_norm=0.1,
    save_interval=1,
    eval_interval=1,
    monitor='semantic.miou',
    monitor_mode='max',
    max_keep_ckpts=5,
    device='cuda',
)