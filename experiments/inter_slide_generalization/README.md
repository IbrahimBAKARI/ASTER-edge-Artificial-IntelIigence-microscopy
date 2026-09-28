# Frozen inter-slide generalization test

The deployed 960-pixel WBC localizer was frozen and evaluated without
retraining, threshold adjustment or model selection on two additional
educational blood-smear slides acquired independently with the same ASTER setup.

| Test set | Fields | Reference WBC | mAP@50 | Precision at 0.18 | Recall at 0.18 |
|---|---:|---:|---:|---:|---:|
| Unseen slide 1 | 79 | 127 | 0.7749 | 0.8652 | 0.6063 |
| Unseen slide 2 | 80 | 114 | 0.9939 | 0.9908 | 0.9474 |
| Pooled | 159 | 241 | 0.9038 | 0.9343 | 0.7676 |

This supports preliminary inter-slide generalization. Two slides do not establish
robustness across laboratories, stains, devices, operators or clinical cohorts.

## Reproduction

Place each slide in a separate folder under:

```text
external_data/unseen_slides/
├── unseen_slide_1/
│   ├── field_001.jpg
│   └── field_001.txt
└── unseen_slide_2/
    ├── field_001.jpg
    └── field_001.txt
```

Each YOLO label row must be `0 x_center y_center width height` with normalized
coordinates. Then run:

```bash
python experiments/inter_slide_generalization/run_inter_slide_evaluation.py \
  --source external_data/unseen_slides \
  --weights models/yolo/wbc_detector.pt \
  --device mps --batch 8
```

The acquired fields and labels are not redistributed; their image-name and
label-content hashes are recorded in `results/provenance.json`. Saved metrics
allow the reported result to be audited without the images.
