"""
DeepStream metadata helpers — batch/frame/object extraction and VLM text extraction.
"""

from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger()

try:
    import pyds  # type: ignore
except ImportError:
    pyds = None  # type: ignore


def get_batch_meta(buffer: Any) -> Optional[Any]:
    """Extract NvDsBatchMeta from a GStreamer buffer."""
    if pyds is None:
        return None
    try:
        return pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
    except Exception as exc:
        logger.debug("get_batch_meta failed", error=str(exc))
        return None


def extract_frame_info(frame_meta: Any) -> Dict[str, Any]:
    """Return a plain dict with frame-level info from NvDsFrameMeta."""
    if pyds is None or frame_meta is None:
        return {}
    try:
        info: Dict[str, Any] = {
            "frame_number": int(frame_meta.frame_num),
            "stream_id": int(frame_meta.pad_index),
            "width": int(frame_meta.source_frame_width),
            "height": int(frame_meta.source_frame_height),
            "timestamp": None,
        }
        for attr in ("pts", "buf_pts", "ntp_timestamp"):
            try:
                info["timestamp"] = int(getattr(frame_meta, attr))
                break
            except (AttributeError, TypeError):
                continue
        return info
    except Exception as exc:
        logger.error("extract_frame_info failed", error=str(exc))
        return {}


def extract_detections(frame_meta: Any) -> List[Dict[str, Any]]:
    """Extract all object detections from NvDsFrameMeta."""
    if pyds is None or frame_meta is None:
        return []
    detections: List[Dict[str, Any]] = []
    l_obj = frame_meta.obj_meta_list
    while l_obj is not None:
        try:
            obj = pyds.NvDsObjectMeta.cast(l_obj.data)
            detections.append({
                "class_id": int(obj.class_id),
                "confidence": float(obj.confidence),
                "left": float(obj.rect_params.left),
                "top": float(obj.rect_params.top),
                "width": float(obj.rect_params.width),
                "height": float(obj.rect_params.height),
            })
        except Exception:
            pass
        finally:
            l_obj = l_obj.next
    return detections


def extract_vlm_text(frame_meta: Any, tensor_name: str = "text_output") -> Optional[str]:
    """
    Extract VLM text from a frame's user-meta list (nvinferserver output).

    Uses PyCapsule pointer extraction (same pattern proven in fyp_main).
    """
    if pyds is None or frame_meta is None:
        return None
    try:
        import ctypes
        import numpy as np

        user_meta_list = frame_meta.frame_user_meta_list
        while user_meta_list is not None:
            try:
                user_meta = pyds.NvDsUserMeta.cast(user_meta_list.data)
                if (user_meta and
                        user_meta.base_meta.meta_type ==
                        pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META):
                    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                    # check tensor name
                    tname = ""
                    if hasattr(tensor_meta, "tensor_name"):
                        raw = tensor_meta.tensor_name
                        tname = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                    if tname == tensor_name or not tname:
                        n_layers = getattr(tensor_meta, "num_output_layers", 0)
                        if n_layers > 0 and hasattr(tensor_meta, "output_layers_info"):
                            for i in range(n_layers):
                                layer = tensor_meta.output_layers_info(i)
                                if not (hasattr(layer, "buffer") and layer.buffer):
                                    continue
                                dims = layer.inferDims
                                total = 1
                                for d in range(dims.numDims):
                                    total *= dims.d[d]
                                try:
                                    PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
                                    PyCapsule_GetPointer.argtypes = [
                                        ctypes.py_object, ctypes.c_char_p]
                                    PyCapsule_GetPointer.restype = ctypes.c_void_p
                                    addr = PyCapsule_GetPointer(
                                        ctypes.py_object(layer.buffer), None)
                                    if addr:
                                        raw_bytes = ctypes.string_at(addr, int(total))
                                        text = (np.frombuffer(raw_bytes, dtype=np.uint8)
                                                .tobytes()
                                                .rstrip(b"\x00")
                                                .decode("utf-8", errors="ignore"))
                                        if text:
                                            return text
                                except Exception:
                                    continue
            except Exception:
                pass
            finally:
                user_meta_list = user_meta_list.next
    except Exception as exc:
        logger.debug("extract_vlm_text failed", error=str(exc))
    return None
