# Changelog

## 1.1.0 — 2026-09-28

- Added a controlled 640/960-pixel resolution and adaptation ablation.
- Added progressive Gaussian-blur and color-perturbation evaluation of the
  frozen OOD gate on fixed public cAItomorph bags.
- Added frozen evaluation of the deployed localizer on two entirely unseen
  educational blood-smear slides.
- Clarified that the 104-field prototype split was held out from fine-tuning
  but contributed to candidate-model selection and is therefore a within-slide
  evaluation rather than a fully independent test.
- Added portable experiment notebooks, compact result artefacts, provenance,
  checksums and the adapted-640 control checkpoint.

## 1.0.0 — 2026-09-17

- Initial public release of the deployed pipeline, frozen models, pre-registered
  decision grid, benchmark scripts and measured results.
