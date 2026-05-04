"""
GStreamer element factory — centralised element creation with error handling.
"""

from typing import Any, Dict, Optional

import structlog

from src.constants import DSConstants

logger = structlog.get_logger()


def _gst():
    from src.environment import get_gst
    return get_gst()


def make_element(factory_name: str, element_name: str, props: Optional[Dict[str, Any]] = None) -> Any:
    """Create a named GStreamer element and set properties."""
    Gst = _gst()
    elem = Gst.ElementFactory.make(factory_name, element_name)
    if not elem:
        raise RuntimeError(f"Failed to create GStreamer element: {factory_name} ({element_name})")
    if props:
        for k, v in props.items():
            try:
                elem.set_property(k, v)
            except Exception as exc:
                logger.warning("Could not set property", prop=k, value=v, error=str(exc))
    logger.debug("Element created", factory=factory_name, name=element_name)
    return elem


def make_uridecodebin(name: str, uri: str) -> Any:
    return make_element(DSConstants.ELEMENT_URIDECODEBIN, name, {"uri": uri})


def make_nvstreammux(name: str, batch_size: int, width: int, height: int, gpu_id: int,
                     max_fps_n: int = 1000, max_fps_d: int = 1,
                     push_timeout_ns: int = 40_000_000) -> Any:
    props: Dict[str, Any] = {
        "batch-size": batch_size,
        "width": width,
        "height": height,
        "gpu-id": gpu_id,
        "live-source": True,
        "batched-push-timeout": push_timeout_ns,
        "sync-inputs": False,
        "drop-pipeline-eos": True,
        "nvbuf-memory-type": 0,
        "overall-max-fps-n": max_fps_n,
        "overall-max-fps-d": max_fps_d,
    }
    elem = make_element(DSConstants.ELEMENT_NVSTREAMMUX, name, props)
    # Re-apply FPS props explicitly (some DS versions need this)
    try:
        elem.set_property("overall-max-fps-n", max_fps_n)
        elem.set_property("overall-max-fps-d", max_fps_d)
    except Exception:
        pass
    return elem


def make_nvinfer(name: str, config_path: str, batch_size: int, gpu_id: int) -> Any:
    return make_element(DSConstants.ELEMENT_NVINFER, name, {
        "config-file-path": config_path,
        "batch-size": batch_size,
        "gpu-id": gpu_id,
    })


def make_nvinferserver(name: str, config_path: str, gpu_id: int,
                       infer_interval: Optional[int] = None) -> Any:
    props: Dict[str, Any] = {"config-file-path": config_path, "gpu-id": gpu_id}
    if infer_interval is not None:
        props["infer-interval"] = infer_interval
    return make_element(DSConstants.ELEMENT_NVINFERSERVER, name, props)


def make_nvvideoconvert(name: str, gpu_id: int) -> Any:
    return make_element(DSConstants.ELEMENT_NVVIDEOCONVERT, name, {"gpu-id": gpu_id})


def make_nvstreamdemux(name: str) -> Any:
    return make_element(DSConstants.ELEMENT_NVSTREAMDEMUX, name)


def make_nvdsosd(name: str, gpu_id: int) -> Any:
    return make_element(DSConstants.ELEMENT_NVDSOSD, name, {"gpu-id": gpu_id})


def make_nvjpegenc(name: str, quality: int = 95) -> Any:
    return make_element(DSConstants.ELEMENT_NVJPEGENC, name, {"quality": quality})


def make_multifilesink(name: str, location: str) -> Any:
    return make_element(DSConstants.ELEMENT_MULTIFILESINK, name, {
        "location": location,
        "sync": False,
        "async": False,
    })


def make_fakesink(name: str) -> Any:
    return make_element(DSConstants.ELEMENT_FAKESINK, name, {
        "sync": False,
        "async": False,
        "silent": True,
    })


def make_queue(name: str) -> Any:
    return make_element(DSConstants.ELEMENT_QUEUE, name)
