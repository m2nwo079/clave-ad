"""AnomalyCLIP image score and anomaly map, reproducing the inference of the official test.py.

The official code is not vendored: the caller passes a checkout of the repository at the pinned commit.
Precision follows the official code (fp32 weights, no autocast).
"""

import pickle
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter


def import_official(repo: Path) -> dict:
    """Put the official repository on the import path; returns the import workarounds that were used."""
    used = {}
    # AnomalyCLIP_lib/AnomalyCLIP.py imports thop.profile at module level but never calls it
    sys.modules["thop"] = types.SimpleNamespace(profile=None)
    used["thop"] = "empty stub module"
    try:
        from pkg_resources import packaging
        used["pkg_resources"] = "pkg_resources"
    except ImportError:
        # the official code only calls packaging.version.parse through pkg_resources
        import packaging.version
        sys.modules["pkg_resources"] = types.SimpleNamespace(packaging=packaging)
        used["pkg_resources"] = "packaging (pkg_resources import failed)"
    sys.path.insert(0, str(repo))
    return used


class AnomalyCLIP:
    def __init__(self, checkpoint: Path, settings: dict, device: str, clip_dir: Path) -> None:
        """checkpoint must already match its configured sha256."""
        import AnomalyCLIP_lib
        from prompt_ensemble import AnomalyCLIP_PromptLearner
        from utils import get_transform

        self.lib, self.settings, self.device = AnomalyCLIP_lib, settings, device
        params = {"Prompt_length": settings["n_ctx"], "learnabel_text_embedding_depth": settings["depth"],
                  "learnabel_text_embedding_length": settings["t_n_ctx"]}
        model, _ = AnomalyCLIP_lib.load(settings["backbone"], device=device, design_details=params,
                                        download_root=str(clip_dir))
        model.eval()
        self.preprocess, _ = get_transform(types.SimpleNamespace(image_size=settings["image_size"]))
        learner = AnomalyCLIP_PromptLearner(model.to("cpu"), params)
        # the caller has checked the checkpoint sha256, so a full unpickle is allowed when weights_only fails
        try:
            state = torch.load(checkpoint, weights_only=True)
            self.torch_load = "weights_only=True"
        except pickle.UnpicklingError:
            state = torch.load(checkpoint, weights_only=False)
            self.torch_load = "weights_only=False (weights_only=True failed)"
        learner.load_state_dict(state["prompt_learner"])
        learner.to(device)
        model.to(device)
        model.visual.DAPM_replace(DPAM_layer=settings["dpam_layer"])
        with torch.no_grad():
            prompts, tokenized, compound = learner(cls_id=None)
            text = model.encode_text_learn(prompts, tokenized, compound).float()
            text = torch.stack(torch.chunk(text, dim=0, chunks=2), dim=1)
            self.text = text / text.norm(dim=-1, keepdim=True)
        self.model = model
        self.dtype = str(model.visual.conv1.weight.dtype)

    @torch.no_grad()
    def __call__(self, image: Image.Image) -> tuple[float, np.ndarray]:
        """Image score (anomaly class probability) and the smoothed map at image_size x image_size."""
        s = self.settings
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        features, patches = self.model.encode_image(x, s["features_list"], DPAM_layer=s["dpam_layer"])
        features = features / features.norm(dim=-1, keepdim=True)
        probs = (features @ self.text.permute(0, 2, 1) / 0.07).softmax(-1)
        maps = []
        for patch in patches:
            patch = patch / patch.norm(dim=-1, keepdim=True)
            similarity, _ = self.lib.compute_similarity(patch, self.text[0])
            similarity = self.lib.get_similarity_map(similarity[:, 1:, :], s["image_size"])
            maps.append((similarity[..., 1] + 1 - similarity[..., 0]) / 2.0)
        anomaly_map = torch.stack(maps).sum(dim=0)[0].cpu().numpy()
        return probs[0, 0, 1].item(), gaussian_filter(anomaly_map, sigma=s["sigma"])
