"""
Triton (nvinferserver) config generator.

Renders the pbtxt config from the embedded template and caches it
in data/cache/triton_configs/ so the same endpoint always reuses
the same file (keyed by a SHA-1 hash of the parameters).
"""

import hashlib
from pathlib import Path
from string import Template
from typing import Optional

import structlog

from src.constants import PathCfg

logger = structlog.get_logger()

_DEFAULT_TEMPLATE = """\
infer_config {
  unique_id: ${UNIQUE_ID}
  max_batch_size: ${MAX_BATCH_SIZE}
  backend {
    triton {
      model_name: "${MODEL_NAME}"
      version: ${MODEL_VERSION}
      grpc {
        url: "${GRPC_URL}"
        enable_cuda_buffer_sharing: ${ENABLE_CUDA}
      }
    }
    inputs { name: "${INPUT_TENSOR}" }
    output_mem_type: MEMORY_TYPE_CPU
  }
  preprocess {
    network_format: IMAGE_FORMAT_RGB
    tensor_order: TENSOR_ORDER_LINEAR
    maintain_aspect_ratio: 0
    tensor_name: "${INPUT_TENSOR}"
  }
  postprocess { other {} }
  extra { copy_input_to_host_buffers: false }
}
input_control { process_mode: PROCESS_MODE_FULL_FRAME }
output_control { output_tensor_meta: true }
"""


def generate_triton_config(
    grpc_endpoint: str,
    model_name: str,
    model_version: int = -1,
    batch_size: int = 1,
    input_tensor: str = "image_input",
    output_tensor: str = "text_output",
    cache_dir: str = PathCfg.DATA_TRITON_DIR,
    template_path: Optional[str] = PathCfg.TRITON_TEMPLATE,
) -> str:
    """
    Render a Triton nvinferserver pbtxt config and cache it to disk.

    Returns the absolute path to the generated config file.
    """
    # Load template (fall back to embedded default)
    template_text = _DEFAULT_TEMPLATE
    if template_path:
        p = Path(template_path)
        if p.exists():
            template_text = p.read_text()

    host, port = _parse_endpoint(grpc_endpoint)
    grpc_url = f"{host}:{port}"

    rendered = Template(template_text).safe_substitute({
        "UNIQUE_ID": 1,
        "MAX_BATCH_SIZE": batch_size,
        "MODEL_NAME": model_name,
        "MODEL_VERSION": model_version,
        "GRPC_URL": grpc_url,
        "ENABLE_CUDA": "true",
        "INPUT_TENSOR": input_tensor,
        "OUTPUT_TENSOR": output_tensor,
    })

    # Cache path keyed by config parameters
    key = hashlib.sha1(
        f"{grpc_url}-{model_name}-{model_version}-{input_tensor}-{output_tensor}".encode()
    ).hexdigest()[:10]
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    config_path = cache / f"triton_vlm_{key}.pbtxt"
    config_path.write_text(rendered)

    logger.info("Triton config generated", path=str(config_path), endpoint=grpc_url,
                model=model_name)
    return str(config_path)


def _parse_endpoint(endpoint: str):
    host, port = endpoint, 8001
    if ":" in endpoint:
        host, port_str = endpoint.rsplit(":", 1)
        try:
            port = int(port_str)
        except ValueError:
            port = 8001
    return host, port


def normalize_model_version(version: Optional[str]) -> int:
    if version in (None, "", "latest"):
        return -1
    try:
        return int(version)
    except (TypeError, ValueError):
        return -1
