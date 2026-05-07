#!/bin/bash
set -e
cd /workspace

echo "=== [0/4] Installing DeepStream Python Bindings ==="
if python3 -c "import pyds" 2>/dev/null; then
    echo "    pyds already installed."
else
    echo "    pyds not found. Installing..."
    apt-get update -q && apt-get install -y -q python3-gi python3-dev python3-gst-1.0
    # Download official bindings for DeepStream 8.0 (v1.2.2)
    echo "    Downloading pyds v1.2.2..."
    WHEEL_URL="https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.2/pyds-1.2.2-cp312-cp312-linux_x86_64.whl"
    pip install -q --break-system-packages "$WHEEL_URL"
fi

echo ""
echo "=== [1/4] Installing Python dependencies ==="
python3 -m pip install -q --break-system-packages -r requirements.txt

echo ""
echo "=== [2/4] Building custom YOLO parser ==="
PARSER_SO=/opt/nvidia/deepstream/deepstream-8.0/lib/libnvdsinfer_custom_impl_Yolo.so

if [ -f "$PARSER_SO" ]; then
    echo "    Parser .so already exists — skipping build."
else
    if [ ! -d /workspace/DeepStream-Yolo ]; then
        echo "    Cloning DeepStream-Yolo..."
        git clone --depth 1 https://github.com/marcoslucianops/DeepStream-Yolo.git /workspace/DeepStream-Yolo
    fi
    echo "    Building CUDA_VER=12.8 ..."
    CUDA_VER=12.8 make -C /workspace/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
    cp /workspace/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so "$PARSER_SO"
    echo "    Parser installed -> $PARSER_SO"
fi

echo ""
echo "=== [3/4] Creating output directories ==="
mkdir -p /workspace/data/cache/frames /workspace/logs

echo ""
echo "=== [4/4] Starting VisionSentry pipeline ==="
echo "    RTSP  : rtsp://localhost:8554/mystream"
echo "    VLM   : 0.0.0.0:8001"
echo ""

python3 main.py \
    --streams "rtsp://127.0.0.1:8554/mystream" \
    --inference-config configs/yolov11m_infer.txt \
    --enable-frame-saving \
    --output-dir data/cache/frames \
    --vlm-endpoint "0.0.0.0:8001" \
    --vlm-model-name nvinferserver_wrapper \
    --vlm-output-tensor text_bytes \
    --vlm-infer-interval 25 \
    --log-level INFO
