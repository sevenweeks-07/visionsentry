"""
DeepStream metadata helpers — batch/frame/object extraction and VLM text extraction.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import structlog

logger = structlog.get_logger()
pyds = None

def _get_pyds():
    global pyds
    if pyds is None:
        try:
            import pyds as pyds_lib
            pyds = pyds_lib
            logger.info("pyds module imported successfully", file=getattr(pyds, "__file__", "unknown"))
        except ImportError as e:
            logger.error("CRITICAL: Could not import pyds. Ensure the wheel is installed.", error=str(e))
            return None
    return pyds


def get_batch_meta(buffer: Any) -> Optional[Any]:
    """Extract NvDsBatchMeta from a GStreamer buffer."""
    lib = _get_pyds()
    if lib is None:
        return None
    try:
        buf_hash = hash(buffer)
        batch_meta = lib.gst_buffer_get_nvds_batch_meta(buf_hash)
        
        if not hasattr(get_batch_meta, "_logged"):
            if batch_meta is not None:
                logger.info("Metadata success: Batch meta extracted", hash=buf_hash)
            else:
                logger.warning("Metadata failure: Batch meta is NULL", hash=buf_hash)
            setattr(get_batch_meta, "_logged", True)
            
        return batch_meta
    except Exception as exc:
        logger.warning("get_batch_meta exception", error=str(exc))
        return None


def extract_frame_info(frame_meta: Any) -> Dict[str, Any]:
    """Return a plain dict with frame-level info from NvDsFrameMeta."""
    lib = _get_pyds()
    if lib is None or frame_meta is None:
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
    lib = _get_pyds()
    if lib is None or frame_meta is None:
        return []
    detections: List[Dict[str, Any]] = []
    l_obj = frame_meta.obj_meta_list
    while l_obj is not None:
        try:
            obj = lib.NvDsObjectMeta.cast(l_obj.data)
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
    Uses PyCapsule pointer extraction.
    """
    lib = _get_pyds()
    if lib is None or frame_meta is None:
        return None
    try:
        import ctypes
        import numpy as np

        user_meta_list = frame_meta.frame_user_meta_list
        while user_meta_list is not None:
            try:
                user_meta = lib.NvDsUserMeta.cast(user_meta_list.data)
                if (user_meta and
                        user_meta.base_meta.meta_type ==
                        lib.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META):
                    tensor_meta = lib.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
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
                                    PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
                                    PyCapsule_GetPointer.restype = ctypes.c_void_p
                                    addr = PyCapsule_GetPointer(ctypes.py_object(layer.buffer), None)
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
def get_frame_image(buffer: Any, frame_meta: Any) -> Optional[np.ndarray]:
    """
    Extract frame image as a numpy array (BGR) from a GStreamer buffer.
    Requires the buffer to be in RGBA/RGB format (e.g. at OSD sink pad).
    """
    lib = _get_pyds()
    if lib is None or buffer is None or frame_meta is None:
        return None
    try:
        # Get surface (this returns a numpy view)
        owner = hash(buffer)
        surface = lib.get_nvds_buf_surface(owner, frame_meta.batch_id)
        
        # Convert to numpy and copy to ensure we have a stable CPU buffer
        img = np.array(surface, copy=True, order='C')
        
        # Handle format: Convert RGBA -> BGR for OpenCV-based logic
        if img.ndim == 3 and img.shape[2] == 4:
            import cv2
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
        return img
    except Exception as exc:
        logger.debug("get_frame_image failed", error=str(exc))
        return None

def extract_ae_score(frame_meta: Any, gie_id: int = 2) -> float:
    """
    Extract the MSE score from the Autoencoder's TensorRT output meta.
    The model outputs a single float32.
    """
    lib = _get_pyds()
    if lib is None or frame_meta is None:
        return 0.0
    try:
        import ctypes
        import numpy as np

        user_meta_list = frame_meta.frame_user_meta_list
        while user_meta_list is not None:
            try:
                user_meta = lib.NvDsUserMeta.cast(user_meta_list.data)
                if (user_meta and
                        user_meta.base_meta.meta_type ==
                        lib.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META):
                    tensor_meta = lib.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                    
                    # Check if this meta comes from our Autoencoder GIE
                    if getattr(tensor_meta, "unique_id", -1) == gie_id:
                        n_layers = getattr(tensor_meta, "num_output_layers", 0)
                        if n_layers > 0 and hasattr(tensor_meta, "output_layers_info"):
                            layer = tensor_meta.output_layers_info(0) # MSE is in the first (and only) layer
                            if hasattr(layer, "buffer") and layer.buffer:
                                # Get pointer to the float32 result
                                PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
                                PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
                                PyCapsule_GetPointer.restype = ctypes.c_void_p
                                addr = PyCapsule_GetPointer(ctypes.py_object(layer.buffer), None)
                                if addr:
                                    # Read 1 float (4 bytes)
                                    raw_bytes = ctypes.string_at(addr, 4)
                                    score = float(np.frombuffer(raw_bytes, dtype=np.float32)[0])
                                    return score
            except Exception:
                pass
            finally:
                user_meta_list = user_meta_list.next
    except Exception as exc:
        logger.debug("extract_ae_score failed", error=str(exc))
    return 0.0
