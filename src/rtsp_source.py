"""
RTSP source wrapper — creates a uridecodebin and wires its dynamic pad
to a requested sink pad on nvstreammux.
"""

from typing import Any, Optional

import structlog

from src.elements import make_uridecodebin, make_queue

logger = structlog.get_logger()


class RTSPSource:
    """One RTSP stream → uridecodebin → nvstreammux."""

    def __init__(self, uri: str, stream_id: int) -> None:
        self.uri = uri
        self.stream_id = stream_id
        self.element: Optional[Any] = None
        self._failed = False

    def create_and_add(self, pipeline: Any) -> bool:
        """Create uridecodebin, add it to pipeline. Returns True on success."""
        try:
            self.element = make_uridecodebin(f"source-{self.stream_id}", self.uri)
            
            # Force TCP (interleaved) mode for stability in Docker
            def _on_source_setup(bin_elem: Any, source: Any) -> None:
                if source.get_factory().get_name() == "rtspsrc":
                    # 4 = GST_RTSP_LOWER_TRANS_TCP
                    source.set_property("protocols", 4)
            
            self.element.connect("source-setup", _on_source_setup)
            
            pipeline.add(self.element)
            return True
        except Exception as exc:
            logger.error("RTSPSource: failed to create element",
                         stream_id=self.stream_id, error=str(exc))
            self._failed = True
            return False

    def connect_to_muxer(self, muxer: Any) -> None:
        """
        Wire the dynamic 'pad-added' signal so decoded video goes into muxer.
        """
        if self._failed or self.element is None:
            return

        def _on_pad_added(src_elem: Any, new_pad: Any, data: Any) -> None:
            caps = new_pad.get_current_caps()
            if caps is None:
                return
            struct = caps.get_structure(0)
            if struct is None:
                return
            name = struct.get_name()
            if not name.startswith("video/x-raw"):
                return
            sink_pad_name = f"sink_{self.stream_id}"
            sink_pad = muxer.get_request_pad(sink_pad_name)
            if sink_pad and not sink_pad.is_linked():
                ret = new_pad.link(sink_pad)
                from src.environment import get_gst
                Gst = get_gst()
                if ret == Gst.PadLinkReturn.OK:
                    logger.info("Source linked to muxer",
                                stream_id=self.stream_id, pad=sink_pad_name)
                else:
                    logger.error("Source→muxer link failed",
                                 stream_id=self.stream_id, result=str(ret))

        self.element.connect("pad-added", _on_pad_added, None)
