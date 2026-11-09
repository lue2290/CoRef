# CoRef

**Open-World Object Detection with Complementary Visual Reference Evidence**

CoRef is a purely visual detector for recognizing known categories and discovering unfamiliar foreground from limited annotations. It reads two complementary forms of evidence from the same labeled reference instances.

- **Reference stage.** Spatial feedback refines candidate locations. A shared mapping and learned pairwise relations compare candidates with individual references. Leave-class episodes teach reference-conditioned rejection.
- **Instance support.** A class-normalized boundary in frozen visual features supplies an independent nomination signal on the same candidates.
- **Source-aware ranking.** Known-only calibration aligns rankings from both routes. Box quality and same-class support verify selected unknown predictions.

[中文说明](README_zh.md) · [Installation](docs/INSTALL.md) · [Data](docs/DATA.md) · [Checkpoints](docs/CHECKPOINTS.md) · [Reproduction](docs/REPRODUCING.md)

## Five-domain RWD results

Overall VOC07 AP50 (%). The local rows average seeds 42, 43, and 44 across five domains.

| Method | Task 1 U | Task 1 K | Task 2 PK | Task 2 CK |
|---|---:|---:|---:|---:|
| CODE (OWL-ViT L/14, published) | 21.7 | 40.8 | 43.6 | 36.2 |
| ROI Classifier | 5.5 | 28.1 | 26.8 | 22.7 |
| CoRef (relation-only) | 17.2 | 57.4 | 54.6 | 54.3 |
| **CoRef** | **23.6** | **58.1** | **56.9** | **56.6** |
| CoRef − CODE (AP50 points) | +1.9 | +17.3 | +13.3 | +20.4 |

In the 7,112-image replay, the relation and support routes uniquely nominate 659 and 1,775 unknown targets. CoRef detects 571 and 1,560 of those targets, respectively.

## Repository layout

```text
CoRef/
├── run_vrdd.py             # scene/task checkpoint download, bank extraction, inference and evaluation
├── configs/                # model and controlled-experiment configurations
├── src/refnet/             # detector, reference relations, losses and adaptation engine
├── src/refnet_f1/          # qualified-candidate calibration and reference decoding
├── src/refnet_ablation/    # controlled component ablations and localization analysis
├── src/owl_reference/      # visual-representation transfer study
├── src/current/            # shared progress and runtime helpers
├── versions/              # imported localization/feedback dependencies
├── vendor/                # PF-RPN code and official evaluation implementation, with notices
├── protocols/<scene>/     # locked Task 1 / Task 2 instance and source-group specifications
├── checkpoints/            # small model index; model tensors are downloaded separately
├── results/                # archived formal summary and environment
├── docs/                   # setup, protocol and checkpoint documentation
├── tools/                  # installation/data preparation helpers
├── tests/                  # numerical and release-interface checks
└── provenance/             # source-file checksums and historical experiment provenance
```

## Quick start

1. Follow [installation](docs/INSTALL.md); install a compatible PyTorch/torchvision and MMCV build before the Python requirements.
2. Place the original PF-RPN and DINOv3 backbones as described in [checkpoints](docs/CHECKPOINTS.md).
3. Download the scene/task bundle for the relation-only configuration:

```bash
python run_vrdd.py download-weights
python run_vrdd.py check --scene Aquatic --task task1
```

4. Obtain RWD through its original distributor and follow [the dataset layout](docs/DATA.md).
5. Build a legal fit-reference bank and run inference:

```bash
python run_vrdd.py build-bank --scene Aquatic --task task1 --data assets/rwd --device cuda:0
python run_vrdd.py infer --scene Aquatic --task task1 --image /path/to/image.jpg --device cuda:0
```

Predictions are written to `outputs/<scene>/<task>/predictions.json`; the associated `classes.json` maps label IDs to known categories and `unknown`. Boxes are XYXY coordinates in original-image pixels.

For the official test population:

```bash
python run_vrdd.py evaluate --scene Aquatic --task task1 --data assets/rwd --device cuda:0
```

The evaluator preserves the archived VOC07 AP50 implementation. New-device numerical differences may change borderline candidates. Re-evaluations are written separately and do not overwrite the archived summary.

## Models and tasks

The release provides Task 1 and Task 2 weights for **Aquatic, Aerial, Game, Medical and Surgical** (called Surgery in the paper). Each scene/task has its own model, reference identities and calibration:

```text
checkpoints/<scene>/<task>/model.pt
checkpoints/<scene>/<task>/calibration.json
checkpoints/<scene>/<task>/reference_bank.json
```

The relation-only configuration uses the same weights for known classification and unknown rejection, with a calibrated margin for the identity decision. Task 2 jointly fits previously and currently known categories.

## Reproducing experiments

See [REPRODUCING.md](docs/REPRODUCING.md) for training, reference-decoder comparisons, component ablations, localization, visual feature transfer, candidate diagnostics, and timing. The public scene/task interface is independent of the original workstation paths.

## Release status

The portable runner implements the relation-only configuration. The full CoRef scores above are paper results for the dual-route system. The release validation covers source compilation, CPU numerical tests, and checkpoint checksums.

## License and acknowledgements

Original release code is distributed under the [MIT License](LICENSE). Third-party code, datasets and pretrained models retain their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). DINOv3 and PF-RPN backbone weights and RWD images are not redistributed by this code repository.
