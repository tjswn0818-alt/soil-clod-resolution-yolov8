# Results and analysis code

Result tables and the final analysis scripts behind the reported numbers for YOLOv8m-seg
soil clod (major axis ≥ 4 cm) instance segmentation, trained and evaluated at different image resolutions.

## Folder structure

```
results/                       result CSV files (CC BY 4.0)
  unified/                     SR/MR evaluation summaries (8 seeds, 4 cm criterion)
  results_review_261026/       bootstrap, PR curve, threshold, NMS and detection-level analyses
  fps_sr_matched_260825.csv    inference speed measurement
  dataset_summary.csv          image / annotation counts per split
  training_config.yaml         training hyperparameters of all models
code/                          analysis scripts (AGPL-3.0)
  config.py                    data / weight / output paths (environment variables)
```

Terms used in the tables: **SR** = single-resolution model, **matched** = inference at the training
resolution, **MR** = multi-resolution (mixed-scale) training, `train_res` / `eval_res` = training /
inference image size (px), `_mean` / `_sd` = mean / SD over 8 seeds, **CI2** = two-stage (image × seed)
bootstrap 95% CI, **CI1 / CI_img** = image-level bootstrap 95% CI, `P_gt0` = bootstrap share of differences > 0.

## Result files

| File | What it contains | Main columns |
|---|---|---|
| `dataset_summary.csv` | Images, annotations and ground truths with major axis ≥ 4 cm per split | split, n_images, n_annotations, n_gt_major_ge_4cm |
| `training_config.yaml` | Training settings shared by all models (from the checkpoints' stored arguments, paths removed) | epochs, batch, optimizer, loss_gains, augmentation, sampling, input_resolution |
| `unified/unified_4cm_sr_summary_260914.csv` | Detection metrics of SR matched models (9 resolutions, test/val) | dataset, model, train_res, eval_res, Recall/Precision/F1/mIoU_mean·_sd, FN_per_img, FP_per_img |
| `unified/unified_4cm_perimage_260914.csv` | Per-image counts for every model, resolution and seed | dataset, model, eval_res, seed, file, n_clod_GT, n_clod_Pred, area4cm_*_cm2, ratio_frame_*, TP, FN, FP |
| `unified/unified_4cm_frag_sr_260914.csv` | Fragmentation metrics of SR matched models | count_MAE, count_R2_1to1, ED_MAE_cm, ED_R2_1to1, area_bias_pct, mean_IoU, ED_KS_D, ratioFrame_MAE (each _mean·_sd) |
| `unified/unified_4cm_cross_recall_{test,val}_260914.csv` | 9 × 9 cross-resolution Recall matrix (train × inference), seed mean | model, train_res, eval_128 … eval_1600 |
| `unified/unified_4cm_cross_recall_{test,val}_sd_260914.csv` | Seed SD of the matrices above | same as above |
| `unified/unified_4cm_mr_summary_260914.csv` | Detection metrics of 6 MR schemes at inference 448/512/736/1024 | model, eval_res, Recall/Precision/F1/mIoU_mean·_sd, FP_per_img, eval_in_train_range |
| `unified/unified_4cm_frag_mr_260914.csv` | Fragmentation metrics of the MR models | model, eval_res, count_MAE, ED_R2_1to1, area_bias_pct, … (_mean·_sd) |
| `unified/unified_4cm_fps_260914.csv` | Inference speed with Recall per resolution | train_res, pure_FPS, full_FPS, pure/full_mean_ms·_sd_ms, Recall_test/val_mean·_sd |
| `fps_sr_matched_260825.csv` | Raw latency measurement (seed 0, batch 1, FP32) | res, pure/full_mean_ms, _sd_ms, _med_ms, _p95_ms, pure_FPS, full_FPS |
| `results_review_261026/ap_summary_261026.csv` | AP50, Recall/Precision at fixed thresholds, max F1 | dataset, train_res, AP50_mean·_sd, Recall_t0.25/0.5, Precision_t0.5, F1_max, t_at_F1max |
| `results_review_261026/bootstrap_recall_ci_261026.csv` | Recall and Precision with bootstrap CIs per resolution | Recall_8seed_mean, Recall_CI_img_lo/hi, Recall_CI_2stage_lo/hi, Precision_* |
| `results_review_261026/per_image_counts_conf05_261026.csv` | Per-image counts at confidence 0.5 (bootstrap input) | train_res, seed, file, n_GT, TP, FN, FP |
| `results_review_261026/bootstrap_pairdiff_261026.csv` | Pairwise resolution differences with bootstrap CIs | metric, pair, delta_point, CI_img_lo/hi, CI_2stage_lo/hi, P_gt0_* |
| `results_review_261026/bootstrap_pairdiff_t025_t075_261026.csv` | Same as above at confidence 0.25 and 0.75 | t, metric, pair, delta_point, CI_* |
| `results_review_261026/pr_curve_{test,val}_261026.csv` | PR curves per resolution and seed (threshold 0.01–0.99) | train_res, seed, t, TP, FP, FN, Precision, Recall, F1 |
| `results_review_261026/matched_conf_summary_261026.csv` | Confidence distribution of true positives | n_matched_pooled8, median, q25, q75, frac_ge_0.5, *_P3/_P4/_P5 |
| `results_review_261026/nms_iou_summary_271026.csv` | Metrics at NMS IoU 0.5 / 0.6 / 0.7 | nms_iou, train_res, recall_rank, Recall/Precision_mean·_sd, FN/FP_per_img_mean |
| `results_review_261026/nms_iou_pairdiff_271026.csv` | Differences between resolutions and between NMS IoU values | nms_iou, contrast, metric, diff, CI2_lo/hi, CI2_excl0, CI1_lo/hi |
| `results_review_261026/nms_iou_rank_271026.csv` | Recall rank of each resolution with bootstrap rank CI | nms_iou, train_res, rank_point, rank_CI_lo/hi, P_last |
| `results_review_261026/nms_iou_perimage_271026.csv` | Per-image counts at each NMS IoU | train_res, seed, nms_iou, file, n_gt, TP, FN, FP |
| `results_review_261026/bootstrap_metrics_ci_271026.csv` | Bootstrap CIs of detection and fragmentation metrics | train_res, table, metric, point_8seed_mean, seed_SD, CI2_lo/hi, CI1_img_lo/hi |
| `results_review_261026/bootstrap_metrics_pairdiff_271026.csv` | Pairwise resolution differences of those metrics | pair, table, metric, diff_point, CI2_lo/hi, CI1_img_lo/hi |
| `results_review_261026/bootstrap_mr_vs_sr_271026.csv` | MR − SR differences per inference resolution | eval_res, metric, MR_point, SR_point, diff_MR_minus_SR, CI2_lo/hi |
| `results_review_261026/level_table_summary_261026.csv` | Share of detections from P3 / P4 / P5 heads | train_res, set, P3/P4/P5_pct_mean·_sd, dominant |
| `results_review_261026/gt_level_maxconf_261026.csv` | Per ground-truth max confidence of P3/P4/P5 candidates before NMS | train_res, seed, file, gt_idx, gt_major_px, P3/P4/P5_max, matched_level_conf05 |
| `results_review_261026/gt_level_maxconf_summary_261026.csv` | Summary of the file above per resolution | P3/P4_max_seedmedian_mean·_sd, *_pooled_median, *_q25, *_q75 |
| `results_review_261026/f1_threshold_transfer_seed_271026.csv` | Validation max-F1 threshold applied to the test set, per seed | train_res, seed, t_valF1max, Recall/Precision/F1_test, Recall_test_t05 |
| `results_review_261026/f1_threshold_transfer_summary_271026.csv` | Seed mean of the file above | train_res, t_mean·_sd·_min·_max, Recall_test_mean, Precision_test_mean |

## Figure data

- Fig. 5: `results_review_261026/per_image_counts_conf05_261026.csv` (same values as Table 3)
- Fig. 6: `results_review_261026/pr_curve_test_261026.csv`, `pr_curve_val_261026.csv`, `ap_summary_261026.csv`
- Fig. 7: `unified/unified_4cm_cross_recall_test_260914.csv`, `unified_4cm_cross_recall_val_260914.csv`
- Fig. 8: schematic, no data file
- Fig. 9: `results_review_261026/gt_level_maxconf_261026.csv`
- Fig. 10: `unified/unified_4cm_mr_summary_260914.csv`, `unified_4cm_frag_mr_260914.csv`
- Fig. 11: `unified/unified_4cm_fps_260914.csv`, `unified/unified_4cm_sr_summary_260914.csv`
- Fig. 12, 13: qualitative image results, no data file

## Code

`unified_4cm_260914.py` produces `results/unified/*`; `fps_sr_260825.py` produces the speed file;
`review_infer_261026.py` runs low-confidence inference that the `review_*` scripts analyse.
Paths are set in `code/config.py` through `RES_SWEEP_ROOT` (weights, labels, outputs) and
`PIXEL_DATA_ROOT` (COCO-format test/val sets).
Original images, annotations, model weights and prediction files (.pkl) are not included,
so a full re-run is not possible from this repository alone.

## Environment

Python 3.13.5, PyTorch 2.6.0, Ultralytics 8.4.69 (`requirements.txt`).

## License

`results/` is licensed under CC BY 4.0 and `code/` under AGPL-3.0 (see `results/LICENSE` and `code/LICENSE`).
