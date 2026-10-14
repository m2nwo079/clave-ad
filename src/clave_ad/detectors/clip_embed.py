"""Original CLIP image embedding of type verification crops (configs/main_run/rules.yaml type.embedding).

The AnomalyCLIP repository loader builds the original CLIP when called without design_details (no learned prompts,
no DPAM). Its encode_image returns every token after ln_post and proj; the CLS token is the original image
embedding. Inputs must be exactly input_size, since other sizes rewrite the position embedding in place.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor

from clave_ad.runinfo import file_sha256


class ClipEmbedder:
    def __init__(self, embedding: dict, device: str, clip_dir: Path) -> None:
        """The official repository must already be on the import path (detectors.anomalyclip.import_official)."""
        import AnomalyCLIP_lib

        model, _ = AnomalyCLIP_lib.load(embedding["backbone"], device=device, download_root=str(clip_dir))
        backbone = next(Path(clip_dir).glob("*.pt"))
        if file_sha256(backbone) != embedding["backbone_sha256"]:
            raise ValueError("CLIP backbone does not match backbone_sha256")
        self.model, self.device, self.size = model.eval(), device, embedding["input_size"]
        self.dtype = str(model.visual.conv1.weight.dtype)
        self.transform = Compose([ToTensor(), Normalize(embedding["normalize_mean"], embedding["normalize_std"])])

    @torch.no_grad()
    def __call__(self, crop: Image.Image) -> np.ndarray:
        # squash_bicubic: the whole crop resized to input_size x input_size without cropping
        x = self.transform(crop.resize((self.size, self.size), Image.Resampling.BICUBIC)).unsqueeze(0).to(self.device)
        cls = self.model.encode_image(x)[0, 0].float()
        return (cls / cls.norm()).cpu().numpy()
