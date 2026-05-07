# 🛡️ VisionSentry TheftGuard

> **Intelligent Multi-Stream Surveillance Pipeline with DeepStream 8.0, YOLOv11, VLM descriptions, and Vector Search (RAG).**

VisionSentry TheftGuard is a state-of-the-art surveillance system designed to detect, describe, and retrieve events in real-time. It combines traditional object detection with anomaly detection and Vision-Language Models (VLM) to provide a semantic understanding of security footage.

---

## 🏗️ System Architecture

The pipeline follows a sophisticated "Dual-Gate" architecture to optimize compute resources while ensuring no critical events are missed.

```mermaid
graph TD
    A[RTSP Stream] --> B[DeepStream Pipeline]
    B --> C{Dual Gate}
    
    C -- "Person Detected (YOLOv11)" --> D[Capture Frame]
    C -- "Anomaly Detected (Autoencoder)" --> D
    
    D --> E[VLM Analysis (Qwen2-VL)]
    E --> F[Generate Semantic Log]
    
    F --> G[(Qdrant Vector DB)]
    G --> H[RAG Retrieval TUI]
    
    subgraph "Backend (Triton)"
    E
    end
    
    subgraph "Storage"
    G
    end
```

---

## 🌟 Key Features

1.  **Dual-Gate Trigger**:
    *   **YOLOv11 Person Gate**: High-precision detection of human presence.
    *   **Autoencoder Anomaly Gate**: A lightweight convolutional autoencoder that flags frames with high reconstruction error (MSE), capturing unexpected events even if no person is detected.
2.  **Semantic Description**: Integrated with **Qwen2-VL** (via TensorRT-LLM/Triton) to generate natural language descriptions of captured events (e.g., *"Person picking up a black backpack"*).
3.  **Vector-Based RAG**: Automatically embeds and indexes VLM descriptions into a **Qdrant** database for instant natural language retrieval.
4.  **Instant Search**: A specialized Terminal UI to search for specific items or behaviors (e.g., *"show me anyone touching a wallet"*).

---

## 🚀 Getting Started

### 1. Prerequisites
*   **NVIDIA GPU** (Ampere/Ada recommended for TensorRT-LLM).
*   **Docker** with `nvidia-container-toolkit`.
*   **YOLOv11 Weights**: Place `yolo11m.onnx` and `yolo11m.onnx.data` in the project root.

### 2. Model Preparation
Prepare the `models` directory for the DeepStream YOLO parser:
```bash
mkdir -p models
cp yolo11m.onnx models/yolo11m_person.onnx
cp yolo11m.onnx.data models/yolo11m.onnx.data
echo "person" > models/coco_labels.txt
```

### 3. Start the VLM Backend (Terminal 1)
The VLM runs in a high-performance TensorRT-LLM backend.
```bash
cd tensorrtllm_backend
./vlm_server_2.sh
```
*Wait for "Started GRPCInferenceService at 0.0.0.0:8001" before proceeding.*

### 4. Start the RTSP Stream (Terminal 2)
Use MediaMTX and FFmpeg to simulate a camera feed:
```bash
./mediamtx &
ffmpeg -re -stream_loop -1 -i person_video.mp4 -c copy -f rtsp rtsp://localhost:8554/mystream
```

### 5. Launch the Pipeline (Terminal 3)
```bash
./launch.sh
```
The pipeline will start processing the stream, saving frames to `data/cache/frames/` and indexing events in `qdrant_db/`.

---

## 🔍 Semantic Search (RAG)

Once the pipeline is running, you can search through the captured events using natural language.

```bash
python3 rag_retrieval.py
```
This launches an interactive TUI where you can ask questions like:
*   *"Did anyone take a blue bag?"*
*   *"Find suspicious behavior near the entrance."*
*   *"Show me frames where a person is eyeing the jewelry."*

---

## 📂 Project Structure

*   `main.py`: Entry point for the DeepStream pipeline.
*   `src/`:
    *   `orchestrator.py`: Manages the overall lifecycle and database ingestion.
    *   `ae_gate.py`: Implementation of the Convolutional Autoencoder for anomaly detection.
    *   `gate.py`: Core logic for the Dual-Gate (YOLO vs AE) decision making.
    *   `pipeline.py`: GStreamer pipeline construction using DeepStream components.
    *   `metadata_writer.py`: Handles saving of JPEGs and JSON metadata.
*   `rag_retrieval.py`: The Vector-RAG search engine and Terminal UI.
*   `configs/`:
    *   `yolov11m_infer.txt`: Configuration for the nvinfer YOLO element.
*   `launch.sh`: Automation script to start the Dockerized environment.

---

## 🛠️ Configuration

You can tune the system behavior in `pipeline_entrypoint.sh` or via CLI arguments in `main.py`:
*   `--vlm-infer-interval`: Frequency of VLM analysis (default: every 25 captured frames).
*   `--enable-frame-saving`: Set to false to disable disk writes.
*   `--log-level`: Set to DEBUG for detailed pipeline telemetry.

---

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
