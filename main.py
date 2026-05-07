#!/usr/bin/env python3
"""
VisionSentry — Multi-Stream RTSP Ingestion with AE Gate.

Entry point for the VisionSentry pipeline.

Example
-------
python main.py \\
    --streams rtsp://192.168.1.10:8554/cam0 \\
    --inference-config configs/yolov11m_infer.txt \\
    --vlm-endpoint 0.0.0.0:8001 \\
    --vlm-model-name ensemble \\
    --enable-frame-saving \\
    --output-dir data/cache/frames \\
    --log-level INFO
"""

import argparse
import logging
import os
import sys

import structlog

from src.orchestrator import Orchestrator
from src.constants import PathCfg, PipelineCfg, ModelCfg

# ── Logging setup ────────────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        (
            structlog.processors.JSONRenderer()
            if os.getenv("JSON_LOGS")
            else structlog.dev.ConsoleRenderer()
        ),
    ]
)
logger = structlog.get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="visionsentry",
        description="VisionSentry — DeepStream 8.0 multi-stream pipeline with AE gate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Streams
    p.add_argument("--streams", nargs="+", required=True,
                   metavar="RTSP_URL",
                   help="One or more RTSP stream URLs (space-separated)")

    # Inference
    p.add_argument("--inference-config",
                   default=PathCfg.YOLOV11M_INFER_CFG,
                   help="Path to nvinfer .txt config (default: configs/yolov11m_infer.txt)")

    # Pipeline dimensions
    p.add_argument("--width",  type=int, default=PipelineCfg.DEFAULT_WIDTH,
                   help=f"Muxed frame width  (default: {PipelineCfg.DEFAULT_WIDTH})")
    p.add_argument("--height", type=int, default=PipelineCfg.DEFAULT_HEIGHT,
                   help=f"Muxed frame height (default: {PipelineCfg.DEFAULT_HEIGHT})")
    p.add_argument("--gpu-id", type=int, default=PipelineCfg.DEFAULT_GPU_ID,
                   help=f"GPU ID (default: {PipelineCfg.DEFAULT_GPU_ID})")

    # Output
    p.add_argument("--output-dir", default=PathCfg.DATA_FRAMES_DIR,
                   help=f"Output directory for frames + metadata (default: {PathCfg.DATA_FRAMES_DIR})")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after saving this many frames (default: unlimited)")

    # Visualization / saving
    p.add_argument("--enable-vis",          action="store_true", default=True,
                   help="Enable bounding-box OSD overlay (default: on)")
    p.add_argument("--disable-vis",         action="store_true", default=False,
                   help="Disable bounding-box OSD overlay")
    p.add_argument("--enable-frame-saving", action="store_true", default=False,
                   help="Save JPEG frames to output-dir")

    # YOLO confidence
    p.add_argument("--confidence", type=float, default=ModelCfg.DEFAULT_CONFIDENCE,
                   help=f"YOLO detection confidence threshold (default: {ModelCfg.DEFAULT_CONFIDENCE})")

    # VLM / Triton
    p.add_argument("--vlm-endpoint",    default=None,
                   help="Triton gRPC endpoint host:port (omit to disable VLM)")
    p.add_argument("--vlm-model-name",  default=None,
                   help="Triton model name (required if --vlm-endpoint set)")
    p.add_argument("--vlm-model-version", default=None,
                   help="Triton model version (default: latest)")
    p.add_argument("--vlm-input-tensor",  default="image_input",
                   help="Triton input tensor name (default: image_input)")
    p.add_argument("--vlm-output-tensor", default="text_bytes",
                   help="Triton output tensor name (default: text_bytes)")
    p.add_argument("--vlm-infer-interval", type=int, default=25,
                   help="VLM throttle: send 1 frame every N frames (default: 25)")

    # Logging
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Log level (default: INFO)")

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    enable_vis = args.enable_vis and not args.disable_vis

    orchestrator = Orchestrator(
        stream_urls=args.streams,
        inference_config=args.inference_config,
        output_dir=args.output_dir,
        gpu_id=args.gpu_id,
        width=args.width,
        height=args.height,
        max_frames=args.max_frames,
        enable_vis=enable_vis,
        enable_frame_saving=args.enable_frame_saving,
        yolo_confidence_threshold=args.confidence,
        vlm_grpc_endpoint=args.vlm_endpoint,
        vlm_model_name=args.vlm_model_name,
        vlm_model_version=args.vlm_model_version,
        vlm_input_tensor=args.vlm_input_tensor,
        vlm_output_tensor=args.vlm_output_tensor,
        vlm_infer_interval=args.vlm_infer_interval,
    )

    try:
        logger.info("VisionSentry starting", streams=len(args.streams))
        orchestrator.setup()
        orchestrator.run()       # blocks
        return 0
    except KeyboardInterrupt:
        logger.info("Stopped by user")
        return 0
    except Exception as exc:
        logger.error("Fatal error", error=str(exc), exc_info=True)
        return 1
    finally:
        orchestrator.teardown()


if __name__ == "__main__":
    sys.exit(main())
