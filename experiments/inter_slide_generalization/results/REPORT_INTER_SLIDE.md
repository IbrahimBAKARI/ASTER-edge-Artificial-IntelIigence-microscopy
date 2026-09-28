# Frozen inter-slide generalization test

The historical 960-pixel adapted localizer was frozen and evaluated on two additional slides that were never used for training, validation, threshold selection, or model selection. The model checksum is recorded in `provenance.json`.

| Test set | Fields | Reference WBC | mAP@50 | mAP@50–95 | TP | FP | FN | Precision¹ | Recall¹ | F1¹ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original slide, within-slide evaluation² | 104 | 171 | 0.9823 | 0.6598 | 164 | 19 | 7 | 0.8962 | 0.9591 | 0.9266 |
| unseen_slide_1 | 79 | 127 | 0.7749 | 0.5008 | 77 | 12 | 50 | 0.8652 | 0.6063 | 0.7130 |
| unseen_slide_2 | 80 | 114 | 0.9939 | 0.6909 | 108 | 1 | 6 | 0.9908 | 0.9474 | 0.9686 |
| combined_unseen_slides | 159 | 241 | 0.9038 | 0.6057 | 185 | 13 | 56 | 0.9343 | 0.7676 | 0.8428 |

¹ Fixed deployment operating point: confidence 0.18, NMS IoU 0.50; TP matching IoU ≥ 0.50. The AP metrics are confidence-integrated Ultralytics metrics.

² Historical within-slide result, included only as a comparator; it is not an
external-slide evaluation. Sources: `../resolution_ablation/results/` and
`../../results/leukocyte_yield/threshold_sweep.csv`.

## Result suitable for the manuscript

> The frozen localizer was evaluated without retraining or threshold adjustment on two independently acquired, entirely unseen educational blood-smear slides comprising 159 fields and 241 annotated WBCs. On the pooled external slides, it achieved mAP@50=0.904 and mAP@50–95=0.606. At the prespecified deployment operating point (confidence 0.18; NMS IoU 0.50), precision was 0.934 and recall was 0.768 (185 TP, 13 FP, and 56 FN).

## Interpretation boundary

This experiment tests transfer to two unseen slides acquired with the same ASTER setup. It is evidence of preliminary inter-slide generalization. Two slides remain a limited sample and do not establish robustness across laboratories, staining protocols, devices, or operators.
