visualization = dict(
    enabled=True,
    save_dir="visualizations",
    save_stage="val",
    alpha=0.45,

    save_original=True,
    save_prediction=True,
    save_ground_truth=True,
    save_semantic_prediction=True,

    save_presence_summary=True,
    save_score_heatmaps=True,

    save_clip_argmax_prediction=True,
    save_clip_score_heatmaps=True,

    vis_prob=0.05,
    max_samples_per_epoch=100,
    vis_seed=42,

    image_folder_pattern="image_{image_id:06d}",
    ignore_index=255,
)