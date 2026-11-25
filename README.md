# Results and analysis code

Result tables and the final analysis scripts behind the reported numbers for YOLOv8m-seg
soil clod (major axis ≥ 4 cm) instance segmentation, trained and evaluated at different image resolutions.

## Folder structure

```
results/                       result CSV files (CC BY 4.0)
  unified/                     SR/MR evaluation summaries (8 seeds, 4 cm criterion)
  results_review_261026/       bootstrap, PR curve, threshold, NMS and detection-level analyses
  results_review_r1_261116/    revision analyses: size-filter protocols, mask-union area, checkpoints and
                               convergence, end-to-end latency, threshold recalibration, image and
                               ground-truth facts, tiled inference, bootstrap of additional contrasts
  results_review_r1_aug_261119/ training-augmentation ablation (mosaic / scale off, 512 and 1600 px, 8 seeds)
  results_review_r1_extra_261123/ tiled inference with merged predictions and its latency; MR training ranges
                               on the test, validation and additional test sets
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

Additional terms in `results_review_r1_*`:
**P0** = the evaluation protocol of the paper (ground truths and predictions both filtered at major axis ≥ 4 cm,
then matched); **P1** = matching before the size filter (all predictions matched against all ground truths; recall
counted on ground truths ≥ 4 cm and precision on predictions ≥ 4 cm, so a prediction ≥ 4 cm matched to a smaller
clod is a TP, not an FP); **P2** = P0 with the prediction filter relaxed to 3.6 cm.
`t` = confidence threshold (score > t), `Recall_t001` = recall at the lowest threshold (t = 0.01),
`pair` = `a-b` difference between training resolutions, `best` / `last` = checkpoint with the best validation
fitness / of epoch 100.
Tiled inference: each image is cut into 3 × 3 tiles of half the frame side (50% overlap), each ground truth is
assigned to one tile; **A** (`A_halfinput`) = tiles resized to half the training resolution, so clods have the same
pixel size as in full-frame matched inference; **B** (`B_sameinput`) = tiles resized to the training resolution,
so clods are twice as large in pixels; `full` = the same code with the whole frame as one tile.
Augmentation conditions: **base** = augmentation of `training_config.yaml`; **mosaic0** = mosaic 0;
**scale0** = scale 0; **mosaic0_scale0** = both 0.

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

### Revision analyses (`results/results_review_r1_261116/`)

All files use the SR matched models (8 seeds) and the evaluation of the main tables (IoU 0.5 greedy matching,
ground truths ≥ 4 cm decided on the original image grid, t = 0.5) unless a column says otherwise.
Areas are computed from the masks.

| File | What it contains | Main columns |
|---|---|---|
| `s0_inventory_runs.csv` | Checkpoints and their training run folders (paths relative to the working root) | family, model, seed, ckpt_copy, sha256_head, run_dir, last_pt, results_csv, args_yaml |
| `s0_repro_baseline.csv` | Baseline metrics recomputed from the stored predictions | dataset, train_res, eval_res, Recall_t05, Precision_t05, FP/FN_per_img, AP50, Recall_t001, areaRatio_MAE/bias_pp |
| `s1_seed_metrics.csv` | Recall, Precision, FN/FP per image under P0/P1/P2 at t = 0.25/0.5/0.75, per seed | protocol, dataset, train_res, seed, t, n_GT4, TP_gt, FN, n_pred4, TP_pred, FP, TPpred_on_smallGT, Recall, Precision |
| `s1_summary_t025_050_075.csv` | Seed mean / SD of the file above | protocol, dataset, train_res, t, Recall/Precision/FN_per_img/FP_per_img_mean·_sd |
| `s1_ap50_recall001_seed.csv`, `s1_ap50_recall001_summary.csv` | AP50 and recall at t = 0.01 under each protocol | protocol, dataset, train_res, AP50, Recall_t001 (_mean·_sd) |
| `s1_bootstrap_recall_ci_t05.csv` | Recall at t = 0.5 with bootstrap CIs per resolution and protocol | protocol, train_res, Recall_8seed_mean, CI_2stage_lo/hi, CI_img_lo/hi |
| `s1_bootstrap_pairdiff_t05.csv` | Recall differences between resolutions per protocol | protocol, pair, delta_point, CI_2stage_lo/hi, P_gt0_2stage, CI_img_lo/hi |
| `s1_fig9_miss_decomposition.csv` | Missed ground truths split into no candidate (≥ 0.01) and low-confidence candidate (0.01–0.5) | protocol, dataset, train_res, miss_no_match_ge001_mean, miss_matched_001_05_mean |
| `s1_diag_fn_removed_by_filter_detail.csv`, `s1_diag_fn_removed_by_filter_summary.csv` | P0 false negatives (t = 0.5) that a prediction < 4 cm would have matched, i.e. misses created by the size filter | dataset, train_res, seed, file, gt_major_cm, hit_by_pred_lt4cm; n_FN_8seed, n_FN_with_lt4cm_pred, ratio |
| `s1_P1_pred4_matched_to_smallGT.csv` | Predictions ≥ 4 cm matched to ground truths < 4 cm (P1), per seed | dataset, train_res, seed, t, n_pred4, TP_pred4, TP_pred4_on_GT_lt4cm |
| `s1_score_ties.csv` | Number of predictions whose score equals the threshold exactly | dataset, train_res, seed, t, n_pred_score_eq_t |
| `s2_area_ratio_per_image.csv` | Coarse-clod area share of the frame as sum of masks and as union of masks | dataset, train_res, seed, file, GT_sum_pct, GT_union_pct, Pred_sum_pct, Pred_union_pct |
| `s2_area_ratio_summary.csv` | Overlap between masks and area-ratio MAE / bias (sum vs union) | GT/Pred_overlap_mean·max_pct, MAE_sum_pp, bias_sum_pp, MAE_union_pp, bias_union_pp (_mean·_sd) |
| `s3_best_vs_last_seed.csv`, `s3_best_vs_last_summary.csv` | best.pt vs last.pt (epoch 100) under the same evaluation | dataset, train_res, best_/last_Recall_t05, Precision_t05, AP50, Recall_t001 |
| `s3_convergence_seed.csv`, `s3_convergence_summary.csv` | Best epoch, validation mask mAP50-95 at best / epoch 100 / max, first epoch reaching 95% of the max, change from epoch 90 to 100 | train_res, seed, best_epoch_ckpt, mAP50-95M_at_best/_at_100/_max, first_epoch_ge95pct_max, delta_*_90to100 |
| `s3_convergence_curves_mean8seed.csv` | Validation curves per epoch, mean of 8 seeds | train_res, epoch, val_mask_mAP50-95_mean, val_seg_loss_mean |
| `s4_environment.csv` | Hardware and software used for the latency measurement | item, value |
| `s4_latency_raw.csv`, `s4_latency_summary.csv` | End-to-end latency per image (ms) | train_res, precision, mask_mode, pre/fwd/post/total_mean·_sd·_median, total_FPS |
| `s5_threshold_seed.csv`, `s5_threshold_summary.csv` | Confidence threshold chosen on the validation set by four rules and applied to the test set | protocol, rule, train_res, t_selected, val_recall_at_t, test_Recall, test_Precision, test_FN/FP_per_img |
| `s5_threshold_pairdiff_bootstrap.csv` | Test recall differences between resolutions under each rule | protocol, rule, pair, delta_point, CI_2stage_lo/hi, P_gt0 |
| `s6_image_sizes.csv`, `s6_image_size_summary.csv` | Original image sizes per split | split, file, W, H, long_min/median/max, n_long_lt1600 |
| `s6_gt_edge_touch.csv` | Ground truths touching the image border | split, n_GT_all, n_edge_all, n_GT_ge4cm, n_edge_ge4cm, ratio_* |
| `s6_gt4cm_major_detail.csv`, `s6_gt4cm_major_summary.csv` | Major axis of ground truths ≥ 4 cm | split, file, major_cm, touches_edge; n, min, p10–p90, max, n_4.0_4.4 |
| `s7_tiling_seed.csv`, `s7_tiling_summary.csv` | Tiled inference (A, B) and full-frame inference, with detection heads of true positives | protocol, condition, train_res, infer_size, Recall_t05, Recall_t001, P3/P4/P5_pct_matched_t05 |
| `s7_tiling_matched_conf_t001.csv` | Score and detection head of every matched prediction at t = 0.01 (tiled and full frame) | protocol, condition, dataset, train_res, infer_size, seed, score, level |
| `s7_tiling_recall_ci.csv` | Recall of each tiling condition and its difference with bootstrap CIs | protocol, condition, train_res, infer_size, contrast, Recall_8seed_mean_or_delta, CI_2stage_lo/hi |
| `s7_cross_eval_reference_P0.csv` | Full-frame recall at the input size that gives the same clod pixel size as condition B | dataset, train_res, eval_res, Recall_8seed_mean |
| `s8_tiling_minus_full_ci.csv` | Tile − full-frame recall difference with bootstrap CIs | protocol, condition, train_res, tile_infer_size, Recall_full, Recall_tile, delta_point, CI_2stage_lo/hi |
| `s8_bootstrap_pairdiff_P1ap_r001_last.csv` | Resolution differences (512/736/1024 − 1600) of P1 AP50, P1 recall at t = 0.01 and best / last checkpoint recall | metric, dataset, pair, value_a, value_b, delta_point, CI_2stage_lo/hi, P_gt0_2stage |
| `s8_P1_pred4_on_smallGT_by_res.csv` | Number of P1 predictions ≥ 4 cm matched to ground truths < 4 cm, per resolution | dataset, train_res, t, n_sum_8seed, n_mean_per_seed, frac_of_TP_pred4_mean |
| `s8_train_gt_major_gt70p7cm.csv` | Training annotation whose major axis exceeds the frame diagonal (70.7 cm); labels unchanged | file_name, ann_id, mask_area_cm2, major_cm, minAreaRect_long_cm, max_Feret_cm |

Latency (`s4_*`): seed 0 SR models, batch 1, the 20 test images held in memory, 10 warm-up and 100 timed runs,
FP32 and FP16, on the hardware in `s4_environment.csv`. `total` = pre-processing (resize, colour conversion,
transfer to the GPU) + forward pass + post-processing (NMS at confidence 0.5 and IoU 0.7, mask construction,
copy of masks and boxes to the CPU). `mask_mode`: `input` = masks at the network input size, `native` = masks at
the original image size.

### Augmentation ablation (`results/results_review_r1_aug_261119/`)

YOLOv8m-seg SR models at 512 and 1600 px, seeds 0–7, trained with one change to `training_config.yaml`
(mosaic 0, scale 0, or both 0; all other settings the same). The baseline (`base`) is the SR-512 and SR-1600
models of the main study (seeds 0–7). Evaluation as in the main tables (P0 and P1, t = 0.5, test and val);
bootstrap: seed × image, B = 10,000, two-stage 95% CI.

| File | What it contains | Main columns |
|---|---|---|
| `aug8_seed_metrics.csv` | Per-seed metrics of every condition (P0/P1, test/val) | condition, protocol, dataset, train_res, seed, Recall_t05, Precision_t05, FN/FP_per_img, AP50, Recall_t001, TP_P3/P4/P5_pct, Det4cm_P3/P4/P5_pct |
| `aug8_summary.csv` | Seed mean / SD of the file above | condition, protocol, dataset, train_res, n_seeds, *_mean·_sd |
| `aug8_vs_baseline.csv` | Baseline, condition and condition − baseline per metric | condition, protocol, dataset, train_res, *_base, *_cond, *_cond_minus_base |
| `aug8_bootstrap.csv` | 512 − 1600 recall difference per condition, condition − baseline per resolution, and change of the 512 − 1600 difference against the baseline | protocol, dataset, condition, contrast, value_a, value_b, delta_point, CI_2stage_lo/hi, P_gt0_2stage |
| `aug8_seed_recall_table.csv` | Recall of each seed (0–7) per condition | condition, protocol, dataset, train_res, seed0 … seed7, mean, sd, min |
| `aug8_collapse_count.csv` | Runs with recall < 0.5 | condition, protocol, dataset, train_res, n_recall_lt0.5, seeds_lt0.5 |
| `aug8_convergence_seed.csv`, `aug8_convergence_summary.csv` | Best epoch, first epoch reaching 95% of the max validation mask mAP50-95, change from epoch 90 to 100 | condition, train_res, seed, best_epoch, first_epoch_ge95pct_max, delta_mAP50-95M_90to100, delta_val_seg_loss_90to100 |
| `aug8_args_check.csv`, `aug8_args_diff_detail.csv` | Training arguments of each run compared with the baseline run of the same resolution and seed (path values shortened) | condition, train_res, seed, intended_changed, diff_keys_unexpected; key, baseline, condition_value, kind |
| `train_manifest.csv` | Epochs, training time and augmentation arguments of each ablation run | condition, res, seed, epochs, train_min, aug_args |

`TP_P*_pct` = detection head of true positives at t = 0.5; `Det4cm_P*_pct` = head of all detections with
confidence > 0.5 and major axis ≥ 4 cm. In `aug8_args_diff_detail.csv`, seed-0 baseline runs differ from the
ablation runs only in `save_period`, `plots` and the dataset path (same image and label files).

### Tiled inference with merged predictions and the additional test set (`results/results_review_r1_extra_261123/`)

**Merged tiled inference (`x1_*`).** Unlike `s7_*` (each ground truth assigned to one tile), every image is cut
into 3 × 3 tiles of half the frame side (50% overlap) and all 9 tiles are inferred without ground-truth
information. Predictions with score > 0.5 are kept; predictions with mask pixels within 2 px of an internal tile
border (a tile side that is not a frame edge) are removed; the remaining masks are placed in frame coordinates
and merged by greedy mask NMS (in descending score, a mask is dropped if its IoU with a kept mask is ≥ 0.5);
predictions with major axis ≥ 4 cm are then matched with the ground truths as in P0 (IoU 0.5, greedy).
Conditions **A** / **B** are those of `s7_*` (`tile_size` = tile input size, `fullframe_size` = full-frame input
size compared with it: the training resolution for A, twice the training resolution for B). SR models, seeds 0–7,
test and val.

`method` values:
**merge_base** = the merging above; **merge_ios** = NMS on intersection over the smaller mask (≥ 0.5) instead of
IoU; **merge_margin0** = border rule removes only masks touching the internal border (0 px);
**merge_margin4** = border rule at 4 px; **merge_noborder** = no border rule; **fullframe** = the whole frame as
one input at `fullframe_size`, score > 0.5 and major axis ≥ 4 cm, no mask NMS; **fullframe_maskNMS** = fullframe
followed by the same mask NMS.

| File | What it contains | Main columns |
|---|---|---|
| `x1_merge_seed.csv` | Per-seed counts and metrics of every method | method, condition, train_res, tile_size, fullframe_size, dataset, seed, n_img, n_GT4, TP, n_pred, FP, FN, Recall, Precision, FN_per_img, FP_per_img |
| `x1_merge_summary.csv` | merge_base, fullframe and fullframe_maskNMS: seed mean / SD with two-stage 95% CI; recall of the ground-truth-assigned tiling (`s7_tiling_summary.csv`, P0) for comparison | method, condition, train_res, dataset, Recall/Precision/FN_per_img/FP_per_img_mean·_sd·_CI_lo·_CI_hi, s7_GTassign_Recall_mean·_sd, s7_n_GT4_unassigned |
| `x1_merge_sensitivity.csv` | Same as above for merge_ios, merge_margin0, merge_margin4, merge_noborder | same as above |
| `x1_merge_pairdiff_ci.csv` | Paired differences `method − minus` (every method − fullframe; each sensitivity variant − merge_base) with two-stage 95% CI | method, minus, condition, train_res, dataset, d_Recall, d_Recall_CI_lo/hi, d_Precision…, d_FN_per_img…, d_FP_per_img… |
| `x1_baseline_check_{a1,a2,a3,b}.csv` | Checks of the pipeline against earlier results: a1 = full frame from the stored predictions vs `s1_seed_metrics.csv` (P0, t = 0.5); a2 = new seed-0 full-frame inference (A) vs the same values; a3 = new full-frame inference for B vs the cross-resolution recall matrix; b = new tiles evaluated with ground-truth assignment vs `s7_tiling_seed.csv` | check, dataset, train_res, infer_size, seed, TP_or_recall, n_pred, n_GT, ref_*, match |
| `x1_latency_raw.csv` | Latency per image and timed run (ms) | condition, train_res, tile_size, fullframe_size, mode, iter, file, pre_ms, fwd_ms, post_ms, merge_ms, total_ms, n_final |
| `x1_latency_summary.csv` | Mean / SD / median of the file above, FPS = 1000 / total_mean | condition, train_res, mode, net_input, pre/fwd/post/merge/total_mean·_sd·_median, total_FPS, n_final_mean |
| `x1_latency_environment.csv` | Hardware and software of the latency measurement | item, value |
| `x1_latency_equiv.csv` | Number of final predictions in the latency run vs the evaluation (seed 0, test), per image | condition, train_res, mode, n_img, n_img_equal, max_abs_diff |
| `x1_latency_logic_check.csv` | GPU merging of the latency code vs the evaluation merging on the same tile masks (seed 0, test) | condition, train_res, tile_size, n_img, n_img_equal |

Bootstrap in `x1_*`: per dataset, rng 2026, B = 10,000; images resampled, then seeds resampled, metric = mean
over the resampled seeds (two-stage); all methods share the same resamples, so differences are paired.
Point values are seed means; `_sd` = SD over the 8 seeds.

Latency `mode`: **fullframe** = one full-frame input at `fullframe_size` (as `s4_*`, `mask_mode` input);
**tile_batch_b9x1** = the 9 tiles as one batch; **tile_seq_b1x9** = the 9 tiles one at a time (9 runs, summed).
Seed 0, FP32, test images in memory, 10 warm-up and 100 timed runs, NMS at confidence 0.5 and IoU 0.7.
`merge` = border rule, mask NMS, transfer of the kept masks to the CPU and the 4 cm filter (tile modes only);
`total` = pre + fwd + post + merge.

**MR training ranges (`x2_*`).** The six MR schemes at inference 448/512/736/1024 and SR models at the same
resolutions (matched; SR-736 = model trained at 720 px, inferred at 736 px), plus SR-1600@1600, seeds 0–7,
evaluated under P0 (t = 0.5, IoU 0.5 greedy, 4 cm) on the test set, the validation set and an additional test
set. Dataset labels: `test`, `val`, and `new20_260623` / `new20_*` = **additional test set**: 20 images taken at
the same site and on the same day as the test set, with 121 clods ≥ 4 cm, not used for training, validation or
the choice of the MR ranges. Its images are not included.

| File | What it contains | Main columns |
|---|---|---|
| `x2_mr_seed.csv` | Additional test set: per-seed metrics of every model and inference resolution | model, family, eval_res, seed, n_img, n_GT, TP, FN, FP, n_pred, Recall, Precision, FN_per_img, FP_per_img, ED_KS_D, n_pred_raw_conf0.001 |
| `x2_mr_summary.csv` | Additional test set: seed mean / SD of the file above | model, eval_res, eval_in_train_range, n_seeds, Recall/Precision/FN_per_img/FP_per_img/ED_KS_D_mean·_sd·_n |
| `x2_mr_compare.csv` | Recall mean / SD, FN and FP per image and KS D side by side: additional test set (`x2_mr_summary.csv`), test and val (`unified/unified_4cm_{mr,sr}_summary_260914.csv`, `unified_4cm_frag_{mr,sr}_260914.csv`) | model, eval_res, eval_in_train_range, new20_/test_/val_Recall_mean·_sd, _FN_per_img, _FP_per_img, _KS_D |
| `x2_mr_pairdiff_ci.csv` | Paired differences: `main` = MR[448-1024] − SR matched, `aux` = MR[448-736] − MR[448-1024] (each at 448/512/736/1024), `aux_sr1600` = SR-512@512 − SR-1600@1600 | dataset, comparison_type, comparison, eval_res, metric, A_point, B_point, diff_A_minus_B, diff_seed_SD, CI2_lo/hi, P2_gt0, resolved_CI2_excludes0, CI1_img_lo/hi, resolved_CI1_excludes0 |
| `x2_baseline_check.csv` | Test set, seed 0: TP / FP / FN and KS D from this code vs `unified_4cm` | model, eval_res, TP_this, TP_unified, …, KS_D_this, KS_D_unified, match |

Bootstrap in `x2_*` (test and val from `unified/unified_4cm_perimage_260914.csv`): rng 2026, B = 10,000, images
resampled, then seeds; **CI2** = two-stage (image × seed) 95% CI, **CI1_img** = image-only 95% CI (seed mean),
`P2_gt0` = two-stage share of differences > 0, `resolved_*_excludes0` = 1 if the CI excludes 0; seed s of model A
is paired with seed s of model B. `ED_KS_D` = Kolmogorov–Smirnov D between ground-truth and predicted
equivalent diameters (images pooled). `eval_in_train_range` = 1 if the inference resolution lies in the MR
training range.

## Figure data

- Fig. 5: `results_review_261026/per_image_counts_conf05_261026.csv` (same values as Table 3)
- Fig. 6: `results_review_261026/pr_curve_test_261026.csv`, `pr_curve_val_261026.csv`, `ap_summary_261026.csv`
- Fig. 7: `unified/unified_4cm_cross_recall_test_260914.csv`, `unified_4cm_cross_recall_val_260914.csv`
- Fig. 8: schematic, no data file
- Fig. 9: `results_review_261026/gt_level_maxconf_261026.csv`
- Fig. 10: (b) recall of tiled inference with ground-truth-selected tiles
  `results_review_r1_261116/s7_tiling_summary.csv`, with merged predictions
  `results_review_r1_extra_261123/x1_merge_summary.csv` (method merge_base), and full frame
  `unified/unified_4cm_cross_recall_test_260914.csv`; (c) merged − full-frame differences
  `results_review_r1_extra_261123/x1_merge_pairdiff_ci.csv` (method merge_base, minus fullframe) and latency ratio
  `results_review_r1_extra_261123/x1_latency_summary.csv` (total_mean of tile_batch_b9x1 / fullframe)
- Fig. 11: `unified/unified_4cm_mr_summary_260914.csv`, `unified_4cm_frag_mr_260914.csv`
- Fig. 12: `unified/unified_4cm_fps_260914.csv`, `unified/unified_4cm_sr_summary_260914.csv`
- Fig. 13, 14: qualitative image results, no data file

## Table data

- Table 6: tiled inference, `results_review_r1_261116/s7_tiling_summary.csv` (conditions A, B and `full`),
  `s7_cross_eval_reference_P0.csv` (full frame at the same clod pixel size for B), `s8_tiling_minus_full_ci.csv`;
  Recall, merged / Precision, merged / Precision, full frame: `results_review_r1_extra_261123/x1_merge_summary.csv`
  (method merge_base and fullframe)
- Table 8: MR training ranges on the test, additional test and validation sets,
  `results_review_r1_extra_261123/x2_mr_compare.csv` (recall mean ± SD) and `x2_mr_pairdiff_ci.csv`
  (comparison_type main, CI2)

## Data

Images and annotations are attached to the [v1.0 release](../../releases/tag/v1.0) (CC BY 4.0).
The dataset is unchanged in v1.1 and v1.2; the same `soil_clod_dataset_v1.0.zip` is attached to the
[v1.1 release](../../releases/tag/v1.1) and the [v1.2 release](../../releases/tag/v1.2).
The images of the additional test set (`results_review_r1_extra_261123/x2_*`) are not included.

`soil_clod_dataset_v1.0.zip`: the images used in the study, cropped to the inside of the 50 cm sampling frame.

```
data/
  train/  images/ (96 .jpg)  _annotations.coco.json  labels/ (YOLO .txt)
  val/    images/ (24 .jpg)  _annotations.coco.json  labels/ (YOLO .txt)
  test/   images/ (20 .jpg)  _annotations.coco.json
```

| Split | Images | Annotated clods |
|---|---|---|
| train | 96 | 1,974 |
| val | 24 | 456 |
| test | 20 | 523 |

- Labels are instance polygons in COCO segmentation format (`_annotations.coco.json`, pixel coordinates).
  `labels/` holds the same polygons in YOLO segmentation format (normalised), used for training.
- pixel (train/val) and 18 (test) both denote the single class 'clod' (names inherited from the Roboflow projects).

The images contain no EXIF metadata (no GPS information).

## Code

`unified_4cm_260914.py` produces `results/unified/*`; `fps_sr_260825.py` produces the speed file;
`review_infer_261026.py` runs low-confidence inference that the `review_*` scripts analyse.
`r1_s0_inventory_repro.py` … `r1_s8_extra.py` (shared functions in `r1_common.py`; run in order s0 → s8,
s8 after s7) produce `results/results_review_r1_261116/*` from the same predictions.
`aug_a1_train.py` trains the augmentation-ablation models (settings in `aug_train_common.py`) and
`aug_a2_eval.py` then produces `results/results_review_r1_aug_261119/*`.
`r1_x1_*.py` and `r1_x2_*.py` (run after `unified_4cm_260914.py` and the `r1_s*` scripts) produce
`results/results_review_r1_extra_261123/*`: merged tiled inference with `r1_x1_infer.py` → `r1_x1_check.py`
(a1, a2, a3, b) → `r1_x1_eval.py` → `r1_x1_latency.py` → `r1_x1_latency_equiv.py`, `r1_x1_latency_logic.py`
(shared functions in `r1_x1_common.py`); MR training ranges with `r1_x2_infer.py base` → `r1_x2_baseline.py` →
`r1_x2_infer.py main` → `r1_x2_eval.py` (shared functions in `r1_x2_common.py`).
Paths are set in `code/config.py` through `RES_SWEEP_ROOT` (weights, labels, outputs),
`PIXEL_DATA_ROOT` (COCO-format test/val sets) and `EXTRA_TEST_DATA_ROOT` (additional test set, not
distributed); training run folders are expected under
`RES_SWEEP_ROOT` (`runs/`, `outputs_pixel_AB/`), and the output folders of the new scripts are set in `config.py`.
Model weights and prediction files (.pkl) are not included, so a full re-run is not possible.

## Environment

Python 3.13.5, PyTorch 2.6.0, Ultralytics 8.4.69 (`requirements.txt`).

## License

`results/` is licensed under CC BY 4.0 and `code/` under AGPL-3.0 (see `results/LICENSE` and `code/LICENSE`).
