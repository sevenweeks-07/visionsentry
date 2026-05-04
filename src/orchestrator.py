"""
VisionSentry Orchestrator.

Wires together:
  - Environment setup (DeepStream 8.0, Docker)
  - PipelineBuilder (GStreamer pipeline)
  - AutoencoderGate  (per-stream MSE anomaly detector)
  - DualGate         (YOLO OR AE → gate_passed, gate_log.csv)
  - MetadataWriter   (JSON per detection)
  - VLM output       (nvinferserver text → stored in pending_metadata)

Gate logic
──────────
On every frame arriving at the nvinfer src pad:
  1. Extract detections → yolo_flagged = any detection with conf >= threshold
  2. Run AutoencoderGate.process() → ae_flagged, ae_score
  3. DualGate.decide()  → gate_passed, gate_reason
  4. If gate_passed → store in pending_metadata for downstream saving.
  5. If NOT gate_passed → discard (do NOT forward to VLM or disk).
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import structlog

# ── DeepStream path (Docker) ──────────────────────────────────────────────────
_DS_ROOT = "/opt/nvidia/deepstream/deepstream-8.0"
sys.path.insert(0, f"{_DS_ROOT}/lib/python3.10/site-packages")

try:
    import pyds  # type: ignore
except ImportError:
    pyds = None  # type: ignore

from src.constants import PathCfg, ModelCfg, PipelineCfg
from src.environment import setup_environment, init_gstreamer, get_gst, get_glib
from src.pipeline import PipelineBuilder
from src.ae_gate import AutoencoderGate
from src.gate import DualGate

logger = structlog.get_logger()


class Orchestrator:
    """
    Top-level orchestrator for VisionSentry.

    Usage
    -----
    orch = Orchestrator(stream_urls=[...], inference_config="configs/yolov11m_infer.txt")
    orch.setup()
    orch.run()          # blocks until EOS / Ctrl-C
    orch.teardown()
    """

    def __init__(
        self,
        stream_urls: List[str],
        inference_config: str,
        output_dir: str = PathCfg.DATA_FRAMES_DIR,
        gpu_id: int = PipelineCfg.DEFAULT_GPU_ID,
        width: int = PipelineCfg.DEFAULT_WIDTH,
        height: int = PipelineCfg.DEFAULT_HEIGHT,
        max_frames: Optional[int] = None,
        enable_vis: bool = True,
        enable_frame_saving: bool = True,
        yolo_confidence_threshold: float = ModelCfg.DEFAULT_CONFIDENCE,
        # VLM (optional)
        vlm_grpc_endpoint: Optional[str] = None,
        vlm_model_name: Optional[str] = None,
        vlm_model_version: Optional[str] = None,
        vlm_input_tensor: str = "image_input",
        vlm_output_tensor: str = "text_output",
        vlm_infer_interval: int = 25,
    ) -> None:
        self.stream_urls = stream_urls
        self.inference_config = inference_config
        self.output_dir = output_dir
        self.gpu_id = gpu_id
        self.width = width
        self.height = height
        self.max_frames = max_frames
        self.enable_vis = enable_vis
        self.enable_frame_saving = enable_frame_saving
        self.conf_threshold = yolo_confidence_threshold

        self.vlm_endpoint = vlm_grpc_endpoint
        self.vlm_model_name = vlm_model_name
        self.vlm_model_version = vlm_model_version
        self.vlm_input_tensor = vlm_input_tensor
        self.vlm_output_tensor = vlm_output_tensor
        self.vlm_infer_interval = vlm_infer_interval

        # Runtime state
        self.builder: Optional[PipelineBuilder] = None
        self.pipeline: Optional[Any] = None
        self.loop: Optional[Any] = None

        self._ae_gate: Optional[AutoencoderGate] = None
        self._dual_gate: Optional[DualGate] = None

        # pending_metadata: {(frame_num, stream_id): {frame_info, detections, vlm_output, _saved}}
        self._pending: Dict[Tuple[int, int], Dict] = {}
        self._frames_with_metadata: set = set()

        self._frame_count = 0
        self._running = False

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """Configure environment, build pipeline, attach probes."""
        setup_environment()
        init_gstreamer()

        self._ae_gate = AutoencoderGate(gpu_id=self.gpu_id)
        self._dual_gate = DualGate(csv_path=PathCfg.GATE_LOG_CSV)

        self.builder = PipelineBuilder(
            stream_urls=self.stream_urls,
            inference_config=self.inference_config,
            output_dir=self.output_dir,
            gpu_id=self.gpu_id,
            width=self.width,
            height=self.height,
            max_frames=self.max_frames,
            enable_vis=self.enable_vis,
            enable_frame_saving=self.enable_frame_saving,
            vlm_grpc_endpoint=self.vlm_endpoint,
            vlm_model_name=self.vlm_model_name,
            vlm_model_version=self.vlm_model_version,
            vlm_input_tensor=self.vlm_input_tensor,
            vlm_output_tensor=self.vlm_output_tensor,
            vlm_infer_interval=self.vlm_infer_interval,
        )
        # Share state dicts with builder
        self.builder.pending_metadata = self._pending
        self.builder.frames_with_metadata = self._frames_with_metadata

        self.pipeline = self.builder.build()

        # Attach detection probe (runs AE gate + dual gate)
        self.builder.add_detection_probe(self._on_detection)

        # Attach VLM output probe if VLM enabled
        if self.vlm_endpoint and self.vlm_model_name:
            self.builder.add_vlm_output_probe(self._on_vlm_output)

        logger.info("Orchestrator setup complete",
                    streams=len(self.stream_urls),
                    vlm=bool(self.vlm_endpoint and self.vlm_model_name))

    def run(self) -> None:
        """Start the pipeline and block until EOS or KeyboardInterrupt."""
        if self.pipeline is None:
            raise RuntimeError("Call setup() before run()")

        Gst = get_gst()
        GLib = get_glib()

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Pipeline failed to enter PLAYING state")

        self._running = True
        logger.info("Pipeline PLAYING — press Ctrl-C to stop")

        self.loop = GLib.MainLoop()
        try:
            self.loop.run()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.teardown()

    def teardown(self) -> None:
        """Stop pipeline, close gate logger."""
        self._running = False
        if self.pipeline:
            self.pipeline.set_state(get_gst().State.NULL)
        if self._dual_gate:
            self._dual_gate.close()
        logger.info("Orchestrator teardown complete",
                    total_frames=self._frame_count)

    # ─────────────────────────────────────────────────────────────────────────
    # Probe callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def _on_detection(
        self,
        frame_number: int,
        stream_id: int,
        detections: List[Dict],
        max_yolo_conf: float,
    ) -> None:
        """
        Called for every frame on the nvinfer src pad.

        Steps
        -----
        1. yolo_flagged = any detection above confidence threshold.
        2. AE gate → ae_flagged, ae_score.
        3. Dual gate → gate_passed, gate_reason.
        4. If gate_passed: store in pending_metadata.
        5. If NOT gate_passed: skip — do NOT save or forward to VLM.
        """
        self._frame_count += 1
        if self._frame_count == 1:
            logger.info("First frame received — pipeline is flowing")

        # ── YOLO flag ─────────────────────────────────────────────────────
        yolo_flagged = any(
            d["confidence"] >= self.conf_threshold for d in detections
        )

        # ── AE gate ───────────────────────────────────────────────────────
        # We need the raw numpy frame to run the AE.
        # In DeepStream we can only get it via GstBuffer → nvvideoconvert
        # (expensive). Instead we use a lightweight CPU fallback via
        # pyds.get_nvds_buf_surface, available in DS 8.0.
        ae_flagged, ae_score = self._run_ae_gate(stream_id, frame_number)

        # ── Dual gate ─────────────────────────────────────────────────────
        gate_passed, gate_reason = self._dual_gate.decide(
            frame_id=frame_number,
            stream_id=stream_id,
            yolo_flagged=yolo_flagged,
            yolo_conf=max_yolo_conf,
            ae_flagged=ae_flagged,
            ae_score=ae_score,
        )

        if not gate_passed:
            # Discard — do not save, do not forward to VLM
            return

        # ── Store pending metadata (downstream probes will save to disk) ──
        if detections:
            key = (frame_number, stream_id)
            self._pending[key] = {
                "frame_info": {
                    "frame_number": frame_number,
                    "stream_id": stream_id,
                },
                "detections": [d for d in detections
                               if d["confidence"] >= self.conf_threshold],
                "vlm_output": None,
                "_saved": False,
                "_gate_reason": gate_reason,
                "_ae_score": ae_score,
            }
            self._frames_with_metadata.add(key)

            # Cleanup old entries to prevent unbounded growth
            if len(self._pending) > 500:
                old_keys = [k for k, v in self._pending.items() if v.get("_saved")]
                for k in old_keys[:200]:
                    self._pending.pop(k, None)

    def _on_vlm_output(self, frame_number: int, stream_id: int, text: str) -> None:
        """
        Called when VLM text is extracted from nvinferserver output.
        Attaches VLM text to existing pending_metadata entry.
        """
        key = (frame_number, stream_id)
        if key in self._pending:
            self._pending[key]["vlm_output"] = text
            logger.info("VLM output stored",
                        frame=frame_number, stream=stream_id,
                        preview=text[:80] if len(text) > 80 else text)
        else:
            # Frame went through VLM but gate hadn't stored it yet
            # (e.g. no YOLO detections but AE flagged) — create entry
            self._pending[key] = {
                "frame_info": {"frame_number": frame_number, "stream_id": stream_id},
                "detections": [],
                "vlm_output": text,
                "_saved": False,
                "_gate_reason": "AE",
                "_ae_score": 0.0,
            }

    def _run_ae_gate(self, stream_id: int, frame_number: int) -> Tuple[bool, float]:
        """
        Attempt to obtain the current frame as a numpy array and run the AE gate.

        DS 8.0 Docker exposes pyds.get_nvds_buf_surface() for zero-copy
        GPU→CPU surface access. If unavailable we return (False, 0.0)
        so the gate degrades gracefully.
        """
        if self._ae_gate is None:
            return False, 0.0
        try:
            # pyds.get_nvds_buf_surface returns an ndarray (BGR) in DS 8.0+
            import numpy as np
            if pyds is None:
                return False, 0.0
            # This is called from the detection probe which runs on the
            # nvinfer src pad — the surface is still available.
            # Surface index corresponds to stream_id in the current batch.
            surface = pyds.get_nvds_buf_surface(0, stream_id)  # type: ignore
            if surface is None:
                return False, 0.0
            frame_np = np.array(surface, copy=True)  # HxWx4 RGBA
            import cv2
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGBA2BGR)
            return self._ae_gate.process(stream_id, frame_bgr)
        except Exception as exc:
            logger.debug("AE gate surface access failed", error=str(exc),
                         note="Gate will be skipped this frame")
            return False, 0.0
