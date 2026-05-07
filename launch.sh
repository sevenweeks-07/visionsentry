#!/bin/bash
# ============================================================================
# VisionSentry — Launch inside DeepStream 8.0 container
# Usage: ./launch.sh
# ============================================================================
set -e

IMAGE="nvcr.io/nvidia/deepstream:8.0-triton-multiarch"
WORKSPACE="$(cd "$(dirname "$0")" && pwd)"
CONTAINER_NAME="visionsentry_v9"

echo "============================================="
echo "  VisionSentry Launcher"
echo "  Workspace : $WORKSPACE"
echo "  Image     : $IMAGE"
echo "============================================="

# Stop any stale container
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

chmod +x "$WORKSPACE/pipeline_entrypoint.sh"

echo ""
echo "Starting container: $CONTAINER_NAME"
echo ""

docker run -it --rm \
    --gpus all \
    --network host \
    --name "$CONTAINER_NAME" \
    -v "$WORKSPACE":/workspace \
    -w /workspace \
    "$IMAGE" \
    bash /workspace/pipeline_entrypoint.sh
