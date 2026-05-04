"""
Convolutional Autoencoder Gate (AE Gate).

Architecture
------------
Encoder: Conv2d(3,16,3,pad=1) → ReLU → MaxPool2d(2)
         → Conv2d(16,8,3,pad=1) → ReLU → MaxPool2d(2)
Decoder: ConvTranspose2d(8,16,2,stride=2) → ReLU
         → ConvTranspose2d(16,3,2,stride=2) → Sigmoid

Input frames are resized to 128×128 before passing through.
MSE reconstruction error is used as the anomaly score.

Calibration
-----------
During the first CALIBRATION_FRAMES of a new stream the gate collects
MSE scores and computes threshold = mean + 2 * std.
After calibration every frame is flagged if MSE > threshold.

Output tuple per frame
----------------------
(frame_bgr_np, metadata_dict, ae_flagged: bool, ae_score: float)
"""

import collections
from typing import Optional, Tuple, Dict, Any

import numpy as np
import structlog

logger = structlog.get_logger()

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms.functional as TF
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — AE gate will always flag frames")

# ──────────────────────────────────────────────────────────────────────────────
# Model definition
# ──────────────────────────────────────────────────────────────────────────────

RESIZE_H = RESIZE_W = 128
CALIBRATION_FRAMES = 500


class _ConvAutoencoder(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Simple convolutional autoencoder for anomaly detection."""

    def __init__(self) -> None:
        if not _TORCH_AVAILABLE:
            return
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),   # 128×128 → 128×128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # 64×64
            nn.Conv2d(16, 8, kernel_size=3, padding=1),   # 64×64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # 32×32
        )
        # Decoder (mirror of encoder)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2),   # 64×64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),   # 128×128
            nn.Sigmoid(),
        )

    def forward(self, x):  # type: ignore[override]
        return self.decoder(self.encoder(x))


# ──────────────────────────────────────────────────────────────────────────────
# Per-stream gate state
# ──────────────────────────────────────────────────────────────────────────────

class AEGateState:
    """Holds calibration state for one RTSP stream."""

    def __init__(self) -> None:
        self.calibrating: bool = True
        self.calibration_scores: list = []
        self.threshold: float = float("inf")

    def update_calibration(self, mse: float) -> None:
        self.calibration_scores.append(mse)
        if len(self.calibration_scores) >= CALIBRATION_FRAMES:
            arr = np.array(self.calibration_scores)
            self.threshold = float(arr.mean() + 2.0 * arr.std())
            self.calibrating = False
            logger.info(
                "AE gate calibration complete",
                threshold=round(self.threshold, 6),
                n_frames=len(self.calibration_scores),
            )

    @property
    def is_calibrating(self) -> bool:
        return self.calibrating


# ──────────────────────────────────────────────────────────────────────────────
# Main gate class
# ──────────────────────────────────────────────────────────────────────────────

class AutoencoderGate:
    """
    Autoencoder anomaly gate.

    Usage
    -----
    gate = AutoencoderGate(gpu_id=0)
    ae_flagged, ae_score = gate.process(stream_id, frame_bgr_np)

    One model is shared across all streams; per-stream calibration state is
    stored in self._states.
    """

    def __init__(self, gpu_id: int = 0) -> None:
        self._device = self._resolve_device(gpu_id)
        self._model: Optional[Any] = None
        self._states: Dict[int, AEGateState] = collections.defaultdict(AEGateState)

        if _TORCH_AVAILABLE:
            self._model = _ConvAutoencoder().to(self._device).eval()
            logger.info("AutoencoderGate initialised", device=str(self._device))
        else:
            logger.warning("AutoencoderGate: PyTorch unavailable, gate disabled")

    # ------------------------------------------------------------------ public

    def process(
        self,
        stream_id: int,
        frame_bgr: np.ndarray,
    ) -> Tuple[bool, float]:
        """
        Run autoencoder on one frame.

        Returns
        -------
        (ae_flagged, ae_score)
            ae_flagged: True if MSE > calibrated threshold (or still calibrating)
            ae_score:   per-frame MSE reconstruction error
        """
        if not _TORCH_AVAILABLE or self._model is None:
            return True, 0.0

        mse = self._compute_mse(frame_bgr)
        state = self._states[stream_id]

        if state.is_calibrating:
            state.update_calibration(mse)
            # During calibration we never flag (not enough context yet)
            return False, mse

        flagged = mse > state.threshold
        return flagged, mse

    # ----------------------------------------------------------------- private

    @torch.no_grad()
    def _compute_mse(self, frame_bgr: np.ndarray) -> float:
        """Resize, normalise, run AE forward pass, return MSE."""
        import cv2
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (RESIZE_W, RESIZE_H))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div(255.0)
        tensor = tensor.unsqueeze(0).to(self._device)            # [1,3,H,W]
        recon = self._model(tensor)                               # [1,3,H,W]
        mse = float(((recon - tensor) ** 2).mean().item())
        return mse

    @staticmethod
    def _resolve_device(gpu_id: int):
        if not _TORCH_AVAILABLE:
            return None
        if torch.cuda.is_available():
            return torch.device(f"cuda:{gpu_id}")
        logger.warning("CUDA not available for AE gate — using CPU")
        return torch.device("cpu")
