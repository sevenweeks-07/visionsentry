# VisionSentry Pipeline

> **DeepStream 8.0 · Docker · YOLO + Convolutional Autoencoder Dual-Gate**

VisionSentry is a multi-stream RTSP video analytics pipeline built on NVIDIA
DeepStream 8.0.  It runs inside a Docker container and combines a YOLO-based
person detector (nvinfer) with a Convolutional Autoencoder anomaly gate to
filter frames before an optional Vision-Language Model (VLM / Triton).

---

## Table of Contents

1. [Architecture](#architecture)
2. [Project Layout](#project-layout)
3. [Autoencoder Gate](#autoencoder-gate)
4. [Dual Gate Logic](#dual-gate-logic)
5. [Gate Log CSV](#gate-log-csv)
6. [Prerequisites](#prerequisites)
7. [Docker Setup](#docker-setup)
8. [Configuration](#configuration)
9. [Running the Pipeline](#running-the-pipeline)
10. [CLI Reference](#cli-reference)
11. [Outputs](#outputs)
12. [Reproducing Exactly](#reproducing-exactly)

---

## Architecture

```
RTSP streams (1-N)
      │
      ▼
 uridecodebin(s)
      │
      ▼
 nvstreammux  ◄── batch frames from all streams
      │
      ▼
 nvinfer (YOLOv11m)  ◄── primary detection
      │
      ├─── [buffer probe] ─────────────────────────────────────────────┐
      │     1. Extract detections  → yolo_flagged                      │
      │     2. AutoencoderGate     → ae_flagged, ae_score              │
      │     3. DualGate.decide()   → gate_passed → gate_log.csv       │
      │     4. gate_passed?  YES → pending_metadata                    │
      │                      NO  → discard frame                       │
      │                                                                 │
      ▼                                                                 │
 [nvinferserver (VLM, optional)]  ◄── throttle probe (persons only)   │
      │   [src pad probe] → extract VLM text → pending_metadata        │
      │                                                                 │
      ▼                                                                 │
 nvvideoconvert                                                         │
      │                                                                 │
      ▼                                                                 │
 nvstreamdemux                                                          │
      │                                                                 │
  (per stream)                                                          │
 queue → [nvdsosd] → nvvideoconvert → nvjpegenc → multifilesink        │
          [conv probe] writes JSON metadata ◄──────────────────────────┘
          [jpegenc probe] saves JPEG with matching frame number
```

---

## Project Layout

```
visionflow_pipeline/
├── main.py                  # CLI entry point
├── run.sh                   # Convenience shell script (Docker)
├── requirements.txt         # Python dependencies
│
├── configs/
│   ├── yolov11m_infer.txt   # nvinfer config for YOLOv11m
│   └── triton_vlm_template.pbtxt  # Rendered at runtime for nvinferserver
│
├── src/
│   ├── __init__.py
│   ├── constants.py         # All constants (paths, sizes, defaults)
│   ├── environment.py       # DeepStream 8.0 env setup (Docker)
│   ├── elements.py          # GStreamer element factory
│   ├── metadata.py          # pyds metadata extraction helpers
│   ├── metadata_writer.py   # JSON frame metadata writer
│   ├── rtsp_source.py       # uridecodebin + dynamic pad wiring
│   ├── pipeline.py          # GStreamer pipeline builder
│   ├── orchestrator.py      # Top-level orchestrator (setup/run/teardown)
│   ├── ae_gate.py           # Convolutional Autoencoder gate
│   ├── gate.py              # Dual gate (YOLO OR AE) + gate_log.csv
│   └── triton_config.py     # Triton pbtxt renderer + cacher
│
├── data/
│   └── cache/
│       ├── frames/          # Saved JPEG frames + JSON metadata
│       └── triton_configs/  # Auto-generated Triton pbtxt files
│
└── logs/
    └── gate_log.csv         # Per-frame gate decisions
```

---

## Autoencoder Gate

**File:** `src/ae_gate.py`

### Architecture

| Layer | Type | Parameters |
|-------|------|-----------|
| Encoder 1 | Conv2d | in=3, out=16, kernel=3, pad=1 → ReLU |
| Encoder pool | MaxPool2d | kernel=2 (128→64) |
| Encoder 2 | Conv2d | in=16, out=8, kernel=3, pad=1 → ReLU |
| Encoder pool | MaxPool2d | kernel=2 (64→32) |
| Decoder 1 | ConvTranspose2d | in=8, out=16, kernel=2, stride=2 → ReLU |
| Decoder 2 | ConvTranspose2d | in=16, out=3, kernel=2, stride=2 → Sigmoid |

- Input frames are **resized to 128×128** before passing to the AE.
- Anomaly score = **MSE** between input and reconstruction.

### Calibration Phase

During the **first 500 frames** of each stream:

```
threshold = mean(MSE scores) + 2 × std(MSE scores)
```

After calibration, a frame is **flagged** (`ae_flagged = True`) if
`MSE > threshold`.

> During calibration, `ae_flagged` is always `False` so no frames are
> incorrectly discarded while the model is warming up.

---

## Dual Gate Logic

**File:** `src/gate.py`

```
yolo_flagged = any detection confidence ≥ threshold
ae_flagged   = MSE > calibrated threshold (post-calibration)

gate_passed = yolo_flagged OR ae_flagged

gate_reason ∈ { "YOLO" | "AE" | "BOTH" | "SKIP" }
```

- `"SKIP"` → frame discarded, NOT forwarded to VLM or saved to disk.
- All other reasons → frame forwarded and potentially saved.

---

## Gate Log CSV

Written to `logs/gate_log.csv` (one row per frame, every stream):

| Column | Description |
|--------|-------------|
| `frame_id` | DeepStream frame number |
| `stream_id` | Source stream index |
| `yolo_conf` | Highest YOLO detection confidence |
| `ae_score` | AE reconstruction MSE |
| `gate_passed` | `True` / `False` |
| `gate_reason` | `YOLO` / `AE` / `BOTH` / `SKIP` |

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| NVIDIA GPU | Ampere or newer recommended |
| CUDA | 12.x (bundled in Docker image) |
| Docker | 24.x+ with `nvidia-container-toolkit` |
| DeepStream Docker image | `nvcr.io/nvidia/deepstream:8.0-gc-triton-devel` |
| YOLOv11m ONNX weights | Your own trained `.onnx` file |
| COCO labels file | `coco_labels.txt` (80-class or 1-class person) |

---

## Docker Setup

### 1. Pull the DeepStream 8.0 image

```bash
docker pull nvcr.io/nvidia/deepstream:8.0-gc-triton-devel
```

> The `gc-triton-devel` variant includes both GStreamer, pyds, and the
> Triton Inference Server libraries needed for nvinferserver.

### 2. Start the container

```bash
docker run -it --rm \
  --gpus all \
  --network host \
  -v /path/to/visionflow_pipeline:/workspace \
  -v /path/to/your/models:/workspace/models \
  -w /workspace \
  --name visionsentry \
  nvcr.io/nvidia/deepstream:8.0-gc-triton-devel \
  bash
```

> `--network host` allows the container to reach the RTSP cameras and
> Triton server running on the host without NAT overhead.

### 3. Install Python dependencies inside container

```bash
pip install -r requirements.txt
```

> GStreamer (`gi.repository.Gst`), `pyds`, and DeepStream plugins are
> **already present** in the image — do not reinstall them via pip.

---

## Configuration

### `configs/yolov11m_infer.txt`

Update the two paths to match your Docker volume mount:

```ini
onnx-file=/workspace/models/yolo11m_person.onnx
labelfile-path=/workspace/models/coco_labels.txt
```

Set `num-detected-classes=1` for a person-only model, or adjust for COCO.

### Custom YOLO parser (required for YOLOv11)

The config uses `NvDsInferParseYoloCuda` from the
[DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) project.

```bash
# Inside the container
git clone https://github.com/marcoslucianops/DeepStream-Yolo.git
cd DeepStream-Yolo
CUDA_VER=12.8 make -C nvdsinfer_custom_impl_Yolo
cp nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so \
   /opt/nvidia/deepstream/deepstream-8.0/lib/
```

### VLM / Triton (optional)

If you have a Triton server running the VLM ensemble model:

```bash
./run.sh rtsp://<cam_ip>:8554/cam0 <triton_host>:8001
```

If you do **not** have Triton, omit the second argument — the pipeline
will run in YOLO + AE gate mode only.

---

## Running the Pipeline

### Inside the container

```bash
# Minimal — YOLO + AE gate only
python main.py \
    --streams rtsp://192.168.68.138:8554/cam0 \
    --inference-config configs/yolov11m_infer.txt \
    --enable-frame-saving \
    --output-dir data/cache/frames \
    --log-level INFO

# With VLM
python main.py \
    --streams rtsp://192.168.68.138:8554/cam0 \
    --inference-config configs/yolov11m_infer.txt \
    --vlm-endpoint 0.0.0.0:8001 \
    --vlm-model-name ensemble \
    --enable-frame-saving \
    --output-dir data/cache/frames \
    --log-level INFO

# Or use the convenience script
./run.sh rtsp://192.168.68.138:8554/cam0 0.0.0.0:8001
```

### Multiple streams

```bash
python main.py \
    --streams rtsp://cam0:8554/live rtsp://cam1:8554/live \
    --inference-config configs/yolov11m_infer.txt \
    --enable-frame-saving
```

---

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--streams` | *(required)* | Space-separated RTSP URLs |
| `--inference-config` | `configs/yolov11m_infer.txt` | nvinfer config |
| `--width` | 1920 | Muxed frame width |
| `--height` | 1080 | Muxed frame height |
| `--gpu-id` | 0 | GPU index |
| `--output-dir` | `data/cache/frames` | Frame + metadata output |
| `--max-frames` | unlimited | Stop after N saved frames |
| `--enable-frame-saving` | off | Save JPEG + JSON to disk |
| `--disable-vis` | off | Disable bounding-box OSD |
| `--confidence` | 0.5 | YOLO detection threshold |
| `--vlm-endpoint` | *(disabled)* | Triton gRPC host:port |
| `--vlm-model-name` | — | Triton model name |
| `--vlm-model-version` | latest | Triton model version |
| `--vlm-input-tensor` | `image_input` | Input tensor name |
| `--vlm-output-tensor` | `text_output` | Output tensor name |
| `--vlm-infer-interval` | 25 | VLM throttle (1 in N frames) |
| `--log-level` | INFO | DEBUG / INFO / WARNING / ERROR |

---

## Outputs

```
data/cache/frames/
└── stream_0/
    ├── frame_000001.jpg          # JPEG frame (gate-passed only)
    ├── frame_000001_info.json    # Detections + optional VLM text
    ├── frame_000042.jpg
    └── frame_000042_info.json

logs/
└── gate_log.csv                  # Every frame's gate decision
```

### `frame_*_info.json` schema

```json
{
  "frame_info": {
    "frame_number": 42,
    "stream_id": 0,
    "width": 1920,
    "height": 1080,
    "timestamp": 1714800000000
  },
  "detections": [
    {
      "class_id": 0,
      "confidence": 0.87,
      "left": 312.0,
      "top": 105.0,
      "width": 98.0,
      "height": 220.0
    }
  ],
  "num_detections": 1,
  "vlm_output": "A person is walking across the frame carrying a bag."
}
```

---

## Reproducing Exactly

To reproduce this pipeline from scratch on a fresh machine:

```bash
# 1. Clone / copy the visionflow_pipeline folder into your container volume
# 2. Pull the image
docker pull nvcr.io/nvidia/deepstream:8.0-gc-triton-devel

# 3. Start container (adjust paths)
docker run -it --rm --gpus all --network host \
  -v $(pwd)/visionflow_pipeline:/workspace \
  -v /path/to/models:/workspace/models \
  -w /workspace \
  nvcr.io/nvidia/deepstream:8.0-gc-triton-devel bash

# 4. Install Python deps
pip install -r requirements.txt

# 5. Build custom YOLO parser
git clone https://github.com/marcoslucianops/DeepStream-Yolo.git
cd DeepStream-Yolo
CUDA_VER=12.8 make -C nvdsinfer_custom_impl_Yolo
cp nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so \
   /opt/nvidia/deepstream/deepstream-8.0/lib/
cd ..

# 6. Edit configs/yolov11m_infer.txt:
#    Set onnx-file and labelfile-path to your model files

# 7. Run
python main.py \
    --streams rtsp://<your_camera>:8554/live \
    --inference-config configs/yolov11m_infer.txt \
    --enable-frame-saving \
    --log-level INFO
```

> **AE Calibration note:** The autoencoder threshold is computed from the
> first **500 frames** of each stream.  During calibration no frames are
> discarded by the AE gate.  The calibrated threshold is logged when
> calibration completes.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| AE gate uses `pyds.get_nvds_buf_surface()` | Zero-copy GPU surface access — no CPU memcpy overhead |
| Per-stream `AEGateState` | Each camera has independent calibration (avoids cross-stream contamination) |
| `DualGate` rule: OR not AND | Maximises recall — a suspicious frame is saved if *either* model flags it |
| VLM throttle probe: person + every Nth | Prevents Triton overload; VLM only sees high-value frames |
| Two-probe frame saving (conv + jpegenc) | Ensures JPEG filename matches the JSON metadata frame number |
| Gate CSV buffering=1 | Line-buffered CSV for durability if the process is killed |
