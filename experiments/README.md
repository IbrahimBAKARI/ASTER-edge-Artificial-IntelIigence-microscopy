# Post hoc controls added in release 1.1.0

These experiments answer three methodological questions raised after the original
prototype analysis. They are descriptive, post hoc controls; none was used to
fit the frozen block-2 decision grid.

| Directory | Question | Main result |
|---|---|---|
| `resolution_ablation/` | Can the 0.371 to 0.982 localizer gain be explained by evaluating at 960 rather than 640 pixels? | No. Resolution alone added 0.129 mAP@50; adaptation at a fixed 640-pixel evaluation added 0.596. |
| `ood_progressive_degradation/` | Does the OOD gate respond progressively to general image degradation? | Scores and rejection rose with blur and color perturbation, most strongly for blur. |
| `inter_slide_generalization/` | Does the frozen localizer transfer beyond the adaptation slide? | Pooled mAP@50 was 0.904 on 159 fields from two entirely unseen slides. |

The repository includes code, protocols, checksums, compact result files and the
additional 640-pixel checkpoint. It does not redistribute public datasets or
prototype-acquired images. See each experiment README for the expected external
data layout.
