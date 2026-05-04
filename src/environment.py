"""
Environment setup for CUDA, TensorRT, and DeepStream 8.0 (Docker).

Configures LD_LIBRARY_PATH, pyds path, and disables proxy detection.
"""

import os
import sys
from typing import Optional

import structlog

from src.constants import DSConstants

logger = structlog.get_logger()

# DeepStream 8.0 Python bindings live at a different sub-path in the image
_PY_BINDINGS = f"{DSConstants.DEFAULT_ROOT}/lib/python3.10/site-packages"
_PLUGIN_PATH = f"{DSConstants.DEFAULT_ROOT}/{DSConstants.PLUGIN_PATH_SUFFIX}"


def setup_environment(deepstream_root: Optional[str] = None) -> None:
    """
    Configure environment for DeepStream 8.0 inside Docker.

    Actions
    -------
    - Clear proxy env-vars (prevents GStreamer segfaults).
    - Extend GST_PLUGIN_PATH with DeepStream plugin directory.
    - Add pyds to sys.path if not already importable.
    """
    root = deepstream_root or DSConstants.DEFAULT_ROOT
    plugin_path = f"{root}/{DSConstants.PLUGIN_PATH_SUFFIX}"

    # --- disable proxy detection ---
    for var in DSConstants.PROXY_ENV_VARS:
        os.environ.pop(var, None)
    os.environ["no_proxy"] = "*"
    os.environ["NO_PROXY"] = "*"

    # --- GStreamer plugin path ---
    existing_gst = os.environ.get("GST_PLUGIN_PATH", "")
    if plugin_path not in existing_gst:
        os.environ["GST_PLUGIN_PATH"] = (
            f"{plugin_path}:{existing_gst}" if existing_gst else plugin_path
        )

    # --- pyds binding path ---
    bindings = f"{root}/lib/python3.10/site-packages"
    try:
        import pyds  # noqa: F401
    except ImportError:
        if bindings not in sys.path:
            sys.path.insert(0, bindings)

    logger.info("Environment configured", deepstream_root=root, plugin_path=plugin_path)


def get_gst():
    """Return the Gst module (gi.repository.Gst)."""
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
    return Gst


def get_glib():
    """Return the GLib module."""
    from gi.repository import GLib
    return GLib


def init_gstreamer() -> None:
    """Initialize GStreamer. Must be called after setup_environment()."""
    Gst = get_gst()
    if not Gst.init_check(None):
        raise RuntimeError("GStreamer initialization failed")
    logger.info("GStreamer initialized")
