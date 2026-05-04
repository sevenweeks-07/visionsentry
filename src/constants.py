"""
Application-wide constants for VisionSentry pipeline.

All constants organized by domain. No magic numbers elsewhere.
DeepStream 8.0 / Docker environment.
"""

from typing import List, Final


class DSConstants:
    """DeepStream 8.0 SDK element and path constants (Docker)."""

    DEFAULT_VERSION: Final[str] = "8.0"
    DEFAULT_ROOT: Final[str] = "/opt/nvidia/deepstream/deepstream-8.0"
    PLUGIN_PATH_SUFFIX: Final[str] = "lib/gst-plugins"

    ELEMENT_URIDECODEBIN: Final[str] = "uridecodebin"
    ELEMENT_NVSTREAMMUX: Final[str] = "nvstreammux"
    ELEMENT_NVSTREAMDEMUX: Final[str] = "nvstreamdemux"
    ELEMENT_NVINFER: Final[str] = "nvinfer"
    ELEMENT_NVINFERSERVER: Final[str] = "nvinferserver"
    ELEMENT_NVVIDEOCONVERT: Final[str] = "nvvideoconvert"
    ELEMENT_NVDSOSD: Final[str] = "nvdsosd"
    ELEMENT_NVJPEGENC: Final[str] = "nvjpegenc"
    ELEMENT_MULTIFILESINK: Final[str] = "multifilesink"
    ELEMENT_FAKESINK: Final[str] = "fakesink"
    ELEMENT_QUEUE: Final[str] = "queue"

    PROXY_ENV_VARS: Final[List[str]] = [
        "http_proxy", "https_proxy", "all_proxy",
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "GIO_USE_PROXY_RESOLVER",
    ]


class PipelineCfg:
    """Pipeline processing defaults."""

    DEFAULT_BATCH_SIZE: Final[int] = 1
    DEFAULT_WIDTH: Final[int] = 1920
    DEFAULT_HEIGHT: Final[int] = 1080
    DEFAULT_GPU_ID: Final[int] = 0

    MUXER_PUSH_TIMEOUT_NS: Final[int] = 40_000_000   # 40 ms in nanoseconds
    MUXER_MAX_FPS_N: Final[int] = 1000               # numerator → unlimited
    MUXER_MAX_FPS_D: Final[int] = 1                  # denominator


class ModelCfg:
    """Model and inference constants."""

    MODEL_YOLOV11M: Final[str] = "yolov11m"
    MODEL_VLM: Final[str] = "vlm"

    DEFAULT_CONFIDENCE: Final[float] = 0.5
    DEFAULT_TRITON_TIMEOUT_MS: Final[int] = 5000

    # COCO class 0 = person (used by YOLO)
    PERSON_CLASS_ID: Final[int] = 0


class PathCfg:
    """File-system path constants (relative to project root)."""

    DATA_FRAMES_DIR: Final[str] = "data/cache/frames"
    DATA_TRITON_DIR: Final[str] = "data/cache/triton_configs"
    LOGS_DIR: Final[str] = "logs"
    GATE_LOG_CSV: Final[str] = "logs/gate_log.csv"

    YOLOV11M_INFER_CFG: Final[str] = "configs/yolov11m_infer.txt"
    TRITON_TEMPLATE: Final[str] = "configs/triton_vlm_template.pbtxt"
