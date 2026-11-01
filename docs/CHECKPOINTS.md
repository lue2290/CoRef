# Scene/task checkpoints

Download [VRDD_weights.zip](https://github.com/lue2290/VRDD/releases/download/v1.0/VRDD_weights.zip), or run `python run_vrdd.py download-weights`.

| Scene | Task 1 | Task 2 |
|---|---|---|
| Aquatic | `checkpoints/Aquatic/task1/model.pt` | `checkpoints/Aquatic/task2/model.pt` |
| Aerial | `checkpoints/Aerial/task1/model.pt` | `checkpoints/Aerial/task2/model.pt` |
| Game | `checkpoints/Game/task1/model.pt` | `checkpoints/Game/task2/model.pt` |
| Medical | `checkpoints/Medical/task1/model.pt` | `checkpoints/Medical/task2/model.pt` |
| Surgical (Surgery) | `checkpoints/Surgical/task1/model.pt` | `checkpoints/Surgical/task2/model.pt` |

Extract the bundle into the repository root. The index stores individual model SHA256 values; `download.json` stores the archive checksum. Every scene/task also has a matching calibration and reference-identity manifest. Do not swap these files between models.

These are adaptation checkpoints. They **do not include the frozen backbones**. Obtain the matching upstream files and place them at:

```text
assets/models/pfrpn/pf_rpn_swinb_5p_coco_imagenet.pth
assets/models/dinov3_large/model.safetensors
assets/models/dinov3_large/config.json
```

Upstream projects: [PF-RPN](https://github.com/tangqh03/PF-RPN) and [DINOv3](https://github.com/facebookresearch/dinov3). Backbone hashes are recorded in `MAINLINE_PARENT_INDEX.json`. Pretrained model terms apply separately from this repository's license.

Only load checkpoints obtained from the trusted release and verify their hashes. The original experiment format includes Python/PyTorch checkpoint metadata and is loaded with the original checkpoint loader.

The released scene/task snapshots support deployment. Aggregate benchmark results in `results/formal_summary.json` describe the archived evaluation population; do not treat the ten deployment snapshots as a newly computed aggregate benchmark.
