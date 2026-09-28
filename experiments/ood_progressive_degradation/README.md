# Progressive OOD degradation control

The frozen block-2 encoder, Mahalanobis feature statistics and threshold are
evaluated on fixed 200-crop bags from 20 cAItomorph stem-cell donors and 10 AML
patients. The same crops are used at every level of Gaussian blur and combined
desaturation/hue shift. No weight, statistic or threshold is refitted.

## Main result

Maximal blur increased rejection from 1/20 to 20/20 donor bags and from 2/10 to
10/10 AML bags. Maximal color perturbation increased rejection to 9/20 and 5/10,
respectively. Median within-patient Spearman correlations between perturbation
severity and OOD score were 1.000 for both families in donors and 0.943/0.971
for blur/color in AML. Individual trajectories were not uniformly monotonic.

This demonstrates sensitivity to controlled image shifts. It does not show that
the gate identifies a particular camera, microscope or ASTER enclosure.

## Reproduction

1. Obtain cAItomorph from its official repository under its original terms.
2. Place the prepared corpus under `external_data/ood_caitomorph/corpus/`, or
   set `ASTER_OOD_CORPUS` and `ASTER_OOD_MANIFEST`.
3. Use `data/selection_manifest.csv` to reproduce the fixed patient/crop
   selection. Images are not included here.
4. Run `ood_progressive_degradation_mps.ipynb` from the repository root on an
   Apple Silicon Mac with MPS. The frozen model files are read from `models/mil/`.

The executed results, input hashes, protocol and CPU/MPS parity check are under
`results/`. `plot_results.py` recreates the English manuscript figure from
`results/ood_scores.csv` without requiring the source images.
