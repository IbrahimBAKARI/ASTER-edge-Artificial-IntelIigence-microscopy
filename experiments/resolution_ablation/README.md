# Resolution and adaptation control

This Colab experiment separates the effect of evaluation resolution from the
effect of mixed-replay adaptation. Three checkpoints are evaluated with the same
Ultralytics evaluator on the same 104-field within-slide evaluation split at
640 and 960 pixels:

- the public-source checkpoint;
- a new mixed-replay checkpoint adapted at 640 pixels;
- the deployed checkpoint adapted at 960 pixels.

## Main contrasts

| Contrast | mAP@50 change |
|---|---:|
| Source detector, evaluation 640 to 960 | +0.1293 |
| Adaptation at 640, evaluated at 640 | +0.5957 |
| Adapted 960 versus source, evaluated at 960 | +0.4808 |

The control shows that larger evaluation inputs alone cannot explain the
adapted performance. Because the 640- and 960-pixel checkpoints come from
separate training runs, their difference does not isolate training resolution
alone.

`resolution_ablation_colab.ipynb` expects the LeukemiaAttri source material and
the prototype dataset in the layouts documented in the notebook. These images
are not redistributed. `models/wbc_detector_adapted_640.pt` is the resulting
control checkpoint; verify it with `models/SHA256SUMS.txt`. Compact metrics,
training arguments, environment capture and input hashes are under `results/`.
