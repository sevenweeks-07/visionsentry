"""
VisionSentry Orchestrator.
Simple, robust version.
"""

from __future__ import annotations

import os
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
import structlog

from src.constants import PathCfg, ModelCfg, PipelineCfg
from src.environment import setup_environment, init_gstreamer, get_gst, get_glib
from src.pipeline import PipelineBuilder
from src.ae_gate import AutoencoderGate
from src.gate import DualGate
from src.metadata_writer import MetadataWriter
from src import metadata as M
from rag_retrieval import SurveillanceRAG

logger = structlog.get_logger()

class Orchestrator:
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
        vlm_grpc_endpoint: Optional[str] = None,
        vlm_model_name: Optional[str] = None,
        vlm_model_version: Optional[str] = None,
        vlm_input_tensor: str = "image_input",
        vlm_output_tensor: str = "text_output",
        vlm_infer_interval: int = 25,
    ) -> None:
        self.stream_urls = stream_urls
        self.inference_config = inference_config
        self.output_dir = Path(output_dir)
        self.conf_threshold = yolo_confidence_threshold
        self.vlm_interval = vlm_infer_interval
        self.vlm_output_tensor = vlm_output_tensor

        self.builder = PipelineBuilder(
            stream_urls=stream_urls,
            inference_config=inference_config,
            output_dir=self.output_dir,
            gpu_id=gpu_id,
            width=width,
            height=height,
            vlm_grpc_endpoint=vlm_grpc_endpoint,
            vlm_model_name=vlm_model_name,
            vlm_model_version=vlm_model_version,
            vlm_input_tensor=vlm_input_tensor,
            vlm_output_tensor=vlm_output_tensor,
            vlm_infer_interval=vlm_infer_interval,
        )
        self._pending: Dict[Tuple[int, int], Dict] = {}
        self._last_block: Dict[int, int] = {} # stream_id -> last block index
        self._ae = AutoencoderGate(gpu_id=gpu_id)
        self._gate = DualGate(csv_path=PathCfg.GATE_LOG_CSV)
        self._frame_count = 0
        
        # Initialize RAG retrieval system
        try:
            self._rag = SurveillanceRAG()
            logger.info("RAG System Initialized")
        except Exception as e:
            logger.error("Failed to initialize RAG system", error=str(e))
            self._rag = None

    def setup(self) -> None:
        setup_environment()
        init_gstreamer()
        self.pipeline = self.builder.build()
        self.builder.add_detection_probe(self._on_detection)
        if self.builder.vlm_endpoint:
            self.builder.add_vlm_output_probe(self._on_vlm_output)

    def run(self) -> None:
        Gst = get_gst(); GLib = get_glib()
        self.pipeline.set_state(Gst.State.PLAYING)
        logger.info("Pipeline PLAYING")
        self.loop = GLib.MainLoop()
        try: self.loop.run()
        except KeyboardInterrupt: pass
        finally: self.pipeline.set_state(Gst.State.NULL)

    def teardown(self) -> None:
        """Cleanup resources."""
        if hasattr(self, "pipeline") and self.pipeline:
            self.pipeline.set_state(get_gst().State.NULL)

    def _on_detection(self, buffer, frame_meta, detections, max_conf):
        frame_num = frame_meta.frame_num
        stream_id = frame_meta.pad_index
        self._frame_count += 1
        if self._frame_count % 100 == 0:
            logger.info("Heartbeat", total=self._frame_count, frame=frame_num)

        # 1. Basic check
        is_person = any(d["confidence"] >= self.conf_threshold for d in detections)
        is_interval = (frame_num % self.vlm_interval == 0)
        
        if not (is_person or is_interval):
            return 

        # 2. Get hardware AE score (extracted from metadata tensor)
        ae_score = M.extract_ae_score(frame_meta)
        ae_flagged, _ = self._ae.process_score(stream_id, ae_score)
        
        # 3. Decision
        passed, reason = self._gate.decide(frame_num, stream_id, is_person, max_conf, ae_flagged, ae_score)
        
        if passed:
            block_idx = frame_num // self.vlm_interval
            if block_idx > self._last_block.get(stream_id, -1):
                self._last_block[stream_id] = block_idx
                key = (frame_num, stream_id)
                # Extract timestamps
                ntp_ts = frame_meta.ntp_timestamp
                # Fallback to buffer PTS if NTP is not available
                pts = frame_meta.buf_pts
                
                info = {
                    "frame_number": frame_num, 
                    "stream_id": stream_id, 
                    "ntp_timestamp": ntp_ts,
                    "pts": pts,
                    "timestamp_utc": datetime.datetime.fromtimestamp(ntp_ts/1e9).isoformat() if ntp_ts > 0 else None
                }
                self._pending[key] = {"detections": detections, "ae_score": ae_score, "info": info}
                
                MetadataWriter(self.output_dir).write(info, detections, None, ae_score)
                logger.info("Frame Captured", stream=stream_id, frame=frame_num, reason=reason)

    def _on_vlm_output(self, frame_num, stream_id, text):
        key = (frame_num, stream_id)
        if key in self._pending:
            logger.info("VLM Update", frame=frame_num, text=text[:50])
            info = self._pending[key].get("info", {})
            MetadataWriter(self.output_dir).write(
                info, 
                self._pending[key]["detections"], 
                text, 
                self._pending[key]["ae_score"]
            )
            
            # 4. Ingest into RAG Vector DB
            if self._rag:
                frame_path = self.output_dir / f"stream_{stream_id}" / f"frame_{frame_num:06d}.jpg"
                self._rag.ingest_log(
                    vlm_text=text,
                    timestamp=info.get("timestamp_utc", datetime.datetime.utcnow().isoformat()),
                    camera_id=f"CAM_{stream_id}",
                    frame_path=str(frame_path)
                )
