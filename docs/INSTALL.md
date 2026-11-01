# Installation

Use Linux with a CUDA-capable GPU for the detector. The archived formal environment used Python 3.12.3, PyTorch 2.8.0+cu128 and FP32. The compatible MMCV extension must be built or installed for the chosen PyTorch/CUDA combination; an arbitrary MMCV wheel is not interchangeable.

1. Create a clean environment. Install the PyTorch/torchvision pair for the target GPU from the official PyTorch instructions.
2. Install a compatible **MMCV 2.x with compiled ops** and MMEngine. Verify `import mmcv.ops` before continuing.
3. Install project dependencies:

```bash
python -m pip install -r requirements.txt
python run_vrdd.py --help
```

The vendored PF-RPN MMDetection implementation takes priority on the Python import path. Do not replace it with an unrelated detector implementation. `tools/install_joint_env.py` preserves an existing PyTorch installation and provides the original environment probing/setup utilities; inspect its `--help` before using it.

Keep `assets/models/dinov3_large/config.json` from this repository. Supply the original matching DINOv3 and PF-RPN backbone files as described in CHECKPOINTS.md. Running `check` reports missing backbones without loading the model and verifies available backbone hashes.

The portable interface runs without hard-coded Windows/server paths. CPU is supported explicitly via `--device cpu`, but detector execution is substantially slower. The full adaptation workflow was designed for a GPU with at least 24 GiB of available memory; workload-specific usage must be measured.
