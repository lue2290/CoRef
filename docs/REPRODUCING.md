# Experiment code map

The user-facing inference path is `run_vrdd.py`. The following preserved entries retain the historical experiment protocol rather than silently rewriting its numerical implementation.

| Study | Entry | Core implementation |
|---|---|---|
| Original detector adaptation | `run_reference_net.py` | `src/refnet/` |
| Fixed-weight F0/F1 calibration and evaluation | `run_reference_f1.py` | `src/refnet_f1/` |
| Controlled component ablations | `run_reference_ablation.py` | `src/refnet_ablation/` |
| Compact ablation export | `run_reference_ablation_compact.py` | `src/reference_ablation_export.py` |
| Localization diagnostics | ablation `diagnose` command | `src/refnet_ablation/` |
| Corrected-annotation Game study | `run_reference_game.py` | `src/refnet_game/` |
| OWL visual-representation transfer | `run_owl_reference.py` | `src/owl_reference/` |
| Full-candidate frozen replay diagnostics | `run_reference_f1_candidates.py` | `src/reference_f1_candidates.py` |
| Actual-device latency measurements | `run_reference_f1_timing.py` | `src/reference_f1_timing.py` |

Use each entry's `--help` to inspect commands. Original training starts with CPU tests, data audit and smoke checks:

```bash
python run_reference_net.py test
python run_reference_net.py audit
python run_reference_net.py smoke --device cuda:0
python run_reference_net.py train --scene Aquatic --task task1 --device cuda:0
```

These archived workflows require the complete dataset, original preparation metadata and their generated caches/locks. A source checkout and the ten deployment weights alone are **not** the full historical experiment workspace. In particular, the strict F0/F1 and ablation workflows verify original parent/cache/result bindings. Run portable evaluation for the released deployment checkpoints; do not disable historical integrity checks to make an incomplete archive appear reproducible.

The preserved training implementation uses detached region descriptors for identity learning, freezes both visual backbones, and trains the relation/feedback/refinement modules under the original schedule. Calibration must use the held-out known candidate population rather than formal unknown test labels. Threshold ties remain known; the F1 margin threshold may be negative.

Frozen candidate replay is a diagnostic of the execution device and may differ numerically from stored original predictions. It does not replace the original formal AP. Runtime measurements separate model loading, prepared-input online inference, end-to-end inference and instrumented stage passes; instrumented or I/O-inclusive time must not be relabeled as pure online latency.

## Source identities

Run `python scripts/test_cpu.py` for the portable numerical/release checks. Two archive-integration checks are explicitly skipped unless `--include-archive-tests` is passed; they need the original corrected XML files and the retired comparison-head workspace. They are not counted as passed GPU or full-archive tests.

`provenance/source_manifest.json` contains release Python-file hashes. `MAINLINE_MANIFEST.json` and `MAINLINE_PARENT_INDEX.json` retain original provenance and internal record identifiers. Deployment filenames and README model tables use only scene/task names; original source identifiers are kept machine-readable for auditability.

Original result archives and full candidate dumps are not committed to Git. `results/` includes lightweight archived evidence. Paper typesetting/admin materials, private correspondence and machine-specific cleanup scripts are intentionally outside this public code release.
