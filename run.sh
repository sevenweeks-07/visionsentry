#!/bin/bash
# ============================================================================
# VisionSentry — Run pipeline inside DeepStream 8.0 Docker container
# ============================================================================
# docker pull nvcr.io/nvidia/deepstream:8.0-triton-multiarch
# Usage:
#   ./run.sh [RTSP_URL] [VLM_ENDPOINT]
#
# Defaults:
#   RTSP_URL     = rtsp://192.168.68.138:8554/cam0
#   VLM_ENDPOINT = 0.0.0.0:8001  (set to empty "" to disable VLM)
#
# Run inside container:
#   docker exec -it visionsentry bash -c "cd /workspace && ./run.sh"
# ============================================================================

set -e

RTSP_STREAM="${1:-rtsp://localhost:8554/mystream}"
VLM_ENDPOINT="${2:-0.0.0.0:8001}"

echo "============================================="
echo "  VisionSentry Pipeline  (DeepStream 8.0)"
echo "============================================="
echo "  RTSP Stream   : $RTSP_STREAM"
echo "  VLM Endpoint  : ${VLM_ENDPOINT:-<disabled>}"
echo "  Gate log      : logs/gate_log.csv"
echo "  Output frames : data/cache/frames/"
echo "============================================="
echo ""

# -- Build VLM arguments only when endpoint is given --------------------------
VLM_ARGS=""
if [ -n "$VLM_ENDPOINT" ]; then
    VLM_ARGS="--vlm-endpoint $VLM_ENDPOINT \
              --vlm-model-name ensemble \
              --vlm-input-tensor image_input \
              --vlm-output-tensor text_output \
              --vlm-infer-interval 25"
fi

python main.py \
    --streams "$RTSP_STREAM" \
    --inference-config configs/yolov11m_infer.txt \
    --enable-frame-saving \
    --output-dir data/cache/frames \
    --log-level INFO \
    $VLM_ARGS

echo ""
echo "============================================="
echo "Pipeline finished."
echo ""
echo "Outputs:"
echo "  Frame JPEGs  → data/cache/frames/stream_0/frame_*.jpg"
echo "  Metadata     → data/cache/frames/stream_0/frame_*_info.json"
echo "  Gate log     → logs/gate_log.csv"
echo "============================================="
