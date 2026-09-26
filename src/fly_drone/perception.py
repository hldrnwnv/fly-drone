"""Optional pretrained visual estimates from consecutive FPV RGB frames.

The depth output is relative inverse depth, not meters. Model predictions are
external sensory estimates; they do not change the MaleCNS connectome weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np


DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
FLOW_MODEL = "torchvision Raft_Small_Weights.C_T_V2"


@dataclass(frozen=True)
class VisualFeatures:
    flow_xy_px: np.ndarray | None
    inverse_depth: np.ndarray
    inference_ms: float
    depth_model: str = DEPTH_MODEL
    flow_model: str = FLOW_MODEL


class NeuralPerception:
    """Run Depth Anything V2 Small and RAFT Small on the FPV camera only."""

    def __init__(self, *, device: str = "cpu", flow_updates: int = 4):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

        self.torch = torch
        self.device = torch.device(device)
        self.flow_updates = flow_updates
        self.depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL)
        self.depth_model = self.depth_model.to(self.device).eval()
        self.flow_weights = Raft_Small_Weights.C_T_V2
        self.flow_model = raft_small(weights=self.flow_weights).to(self.device).eval()
        self.previous_frame: np.ndarray | None = None
        self.history: list[tuple[float, np.ndarray, VisualFeatures]] = []

    def reset(self) -> None:
        self.previous_frame = None
        self.history.clear()

    def observe(self, frame: np.ndarray, *, time_s: float = 0.0) -> VisualFeatures:
        from PIL import Image
        from torchvision.transforms import functional as F

        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("frame must be HxWx3 uint8 RGB")
        started = perf_counter()
        image = Image.fromarray(frame)
        inputs = self.depth_processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            depth_output = self.depth_model(**inputs)
            depth = self.depth_processor.post_process_depth_estimation(
                depth_output, target_sizes=[frame.shape[:2]])[0]["predicted_depth"]
        inverse_depth = depth.float().cpu().numpy().astype(np.float32)
        flow = None
        if self.previous_frame is not None:
            old = self.torch.from_numpy(self.previous_frame.copy()).permute(2, 0, 1)[None]
            new = self.torch.from_numpy(frame.copy()).permute(2, 0, 1)[None]
            old = F.resize(old, [128, 224], antialias=False)
            new = F.resize(new, [128, 224], antialias=False)
            old, new = self.flow_weights.transforms()(old, new)
            with self.torch.inference_mode():
                estimated = self.flow_model(old.to(self.device), new.to(self.device),
                                            num_flow_updates=self.flow_updates)[-1][0]
            flow = estimated.permute(1, 2, 0).float().cpu().numpy().astype(np.float32)
            flow[..., 0] *= frame.shape[1] / 224
            flow[..., 1] *= frame.shape[0] / 128
        self.previous_frame = frame.copy()
        features = VisualFeatures(flow, inverse_depth, 1000 * (perf_counter() - started))
        self.history.append((time_s, frame.copy(), features))
        return features

    def save(self, path) -> None:
        """Persist exact camera frames and model outputs used by one episode."""
        if not self.history:
            raise RuntimeError("No visual observations to save")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            t_s=np.array([item[0] for item in self.history]),
            rgb=np.stack([item[1] for item in self.history]),
            relative_inverse_depth=np.stack([item[2].inverse_depth
                                             for item in self.history]),
            flow_xy_px=np.stack([item[2].flow_xy_px if item[2].flow_xy_px is not None
                                 else np.zeros((128, 224, 2), dtype=np.float32)
                                 for item in self.history]),
            flow_valid=np.array([item[2].flow_xy_px is not None
                                 for item in self.history]),
        )
