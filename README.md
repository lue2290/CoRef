# VRDD

**Visual Reference-Driven Open-World Object Detection**

VRDD recognizes known categories and rejects unknown foreground using labeled visual reference instances. It combines a prompt-free proposal network, frozen DINOv3 features, learned instance relations and calibrated rejection. No task-category text or attribute prompts are required for category verification.

[中文说明](README_zh.md) · [Installation](docs/INSTALL.md) · [Data](docs/DATA.md) · [Checkpoints](docs/CHECKPOINTS.md) · [Reproduction](docs/REPRODUCING.md)

![Visual reference evidence](docs/figures/VRDD_Visual_Reference_Intuition.png)

## Repository layout

```text
VRDD/
├── run_vrdd.py             # scene/task checkpoint download, bank extraction, inference and evaluation
├── configs/                # model and controlled-experiment configurations
├── src/refnet/             # detector, reference relations, losses and adaptation engine
├── src/refnet_f1/          # qualified-candidate calibration and F1 decoding
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
3. Download the VRDD scene/task model bundle:

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

The same weights are used for known classification and unknown rejection; F1 denotes the calibrated decision rule rather than a separate trained network. Task 2 is joint fitting of old and newly introduced known categories, not exemplar-free continual learning.

## Reproducing experiments

See [REPRODUCING.md](docs/REPRODUCING.md) for original training, F0/F1, component ablations, localization, OWL transfer, candidate diagnostics and runtime measurement entry points. Historical source identities are retained; internal source records are not model filenames. The public scene/task interface is independent of the original workstation paths.

## Release status

This repository contains preserved experiment code and a new portable deployment interface. Source compilation, CPU numerical tests and checkpoint checksums are checked locally. End-to-end GPU execution of the portable interface still requires validation in the target detector environment; it is not claimed as newly benchmarked here.

## License and acknowledgements

Original release code is distributed under the [MIT License](LICENSE). Third-party code, datasets and pretrained models retain their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). DINOv3 and PF-RPN backbone weights and RWD images are not redistributed by this code repository.
