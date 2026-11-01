from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image,ImageOps
from torchvision.ops import roi_align
from .protocol import sha as sha256

def expanded(boxes: torch.Tensor, width: float, height: float, factor=1.5):
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    sizes = (boxes[:, 2:] - boxes[:, :2]) * factor / 2
    lo = torch.maximum(centers - sizes, boxes.new_zeros(2))
    hi = torch.minimum(centers + sizes, boxes.new_tensor([width, height]))
    return torch.cat([lo, hi], dim=1)

def pool_spatial(feature_map: torch.Tensor, boxes: torch.Tensor, stride: int,
                 width: float, height: float):
    center = roi_align(feature_map, [boxes], output_size=(2, 2),
                       spatial_scale=1 / stride, sampling_ratio=2, aligned=True)
    context_boxes = expanded(boxes, width, height)
    context = roi_align(feature_map, [context_boxes], output_size=(1, 1),
                        spatial_scale=1 / stride, sampling_ratio=2, aligned=True)
    return torch.cat([center.flatten(1), context.flatten(1)], dim=1)

def dinov3_timm_extractor(model_dir: Path, device: str, max_side: int,
                          architecture: str, precision: str):
    """Load a public timm DINOv3 checkpoint without contacting the Hub."""
    import timm
    from safetensors.torch import load_file

    config_file = model_dir / "config.json"
    weights_file = model_dir / "model.safetensors"
    if not config_file.is_file() or not weights_file.is_file():
        raise FileNotFoundError(f"timm DINOv3 config/weights missing in {model_dir}")
    model_config = json.loads(config_file.read_text(encoding="utf-8"))
    if (model_config.get("architecture") != architecture or
            model_config.get("pretrained_cfg", {}).get("tag") != "lvd1689m"):
        raise ValueError(f"Unexpected timm DINOv3 architecture or pretraining: {config_file}")
    if precision == "fp16" and not str(device).startswith("cuda"):
        raise ValueError("fp16 DINOv3 extraction requires CUDA")
    dtype = torch.float16 if precision == "fp16" else torch.float32
    model = timm.create_model(architecture, pretrained=False, num_classes=0)
    model.load_state_dict(load_file(str(weights_file), device="cpu"), strict=True)
    model = model.to(device=device, dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    patch = int(model.patch_embed.patch_size[0])
    mean = torch.tensor(model_config["pretrained_cfg"]["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(model_config["pretrained_cfg"]["std"], dtype=torch.float32).view(3, 1, 1)
    signature = (sha256(weights_file) + ":" + sha256(config_file) +
                 f":timm-{timm.__version__}:{precision}-full-image-patches-{max_side}-spatial2-context1-v1")

    def extract(image_path, boxes, _old_features):
        with Image.open(image_path) as raw:
            im = ImageOps.exif_transpose(raw).convert("RGB")
            width, height = im.size
            ratio = max_side / max(width, height)
            new_width = max(patch, round(width * ratio / patch) * patch)
            new_height = max(patch, round(height * ratio / patch) * patch)
            im = im.resize((new_width, new_height), Image.Resampling.BICUBIC)
            pixels = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
        pixels = ((pixels - mean) / std).unsqueeze(0).to(device=device, dtype=dtype)
        with torch.inference_mode():
            tokens = model.forward_features(pixels)
            grid_h, grid_w = new_height // patch, new_width // patch
            patch_count = grid_h * grid_w
            if tokens.ndim != 3 or tokens.shape[1] != patch_count + model.num_prefix_tokens:
                raise ValueError(f"Unexpected DINOv3 token shape: {tuple(tokens.shape)}")
            patches = tokens[:, -patch_count:, :]
            feature_map = patches.reshape(1, grid_h, grid_w, -1).permute(0, 3, 1, 2).contiguous()
            mapped = torch.from_numpy(boxes).to(device=device, dtype=feature_map.dtype) * feature_map.new_tensor(
                [new_width / width, new_height / height] * 2)
            pooled = pool_spatial(feature_map, mapped, patch, new_width, new_height)
        return pooled.float().cpu().numpy()

    return extract, signature, None

class Extractor:
 def __init__(self,root,device):
  from mmdet.apis import init_detector
  from mmdet.utils import register_all_modules
  register_all_modules(init_default_scope=True)
  from mmengine.config import Config
  self.root=Path(root);self.device=device
  cfg=Config.fromfile(str(self.root/'vendor/pfrpn/configs/pf-rpn/pf-rpn_coco-imagenet.py'))
  cfg.model.backbone.pop('init_cfg',None);cfg.model.pop('init_cfg',None)
  self.pf=init_detector(cfg,str(self.root/'assets/models/pfrpn/pf_rpn_swinb_5p_coco_imagenet.pth'),device=device,palette='random').eval()
  for p in self.pf.parameters():p.requires_grad_(False)
  arch=json.loads((self.root/'assets/models/dinov3_large/config.json').read_text())['architecture']
  self.dino,self.signature,_=dinov3_timm_extractor(self.root/'assets/models/dinov3_large',device,518,arch,'fp32')
 def __call__(self,scene,name):
  from mmdet.apis import inference_detector
  path=self.root/'assets/rwd/JPEGImages'/scene/name
  with torch.inference_mode():
   result=inference_detector(self.pf,str(path));pred=result.pred_instances;ids=pred.scores.argsort(descending=True,stable=True)[:300];boxes=pred.bboxes[ids].cpu().numpy();scores=pred.scores[ids].cpu().numpy()
  features=self.dino(path,boxes,None)
  return boxes.astype(np.float32),scores.astype(np.float32),features.astype(np.float32)
