# ASTER V1.1.0

This release adds the three post hoc controls reported in the revised IEEE
Access manuscript:

- evaluation resolution alone raises the source localizer from 0.372 to 0.502
  mAP@50, while adaptation at a fixed 640-pixel evaluation raises it to 0.968;
- progressive blur and color perturbations increase the frozen OOD score and
  rejection rate on fixed cAItomorph bags;
- the frozen localizer reaches pooled mAP@50 0.904 on 159 fields from two
  entirely unseen educational smears.

No public dataset or prototype-acquired image is redistributed. The archive
contains code, frozen/deployed models, the adapted-640 control checkpoint,
compact results, manifests and checksums.
