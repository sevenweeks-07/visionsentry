"""
VisionSentry Pipeline Builder.

Assembles the GStreamer pipeline:

  uridecodebin(s) ──► nvstreammux ──► nvinfer (YOLO)
                                           │
                              [nvinferserver (VLM, optional)]
                                           │
                                   nvvideoconvert
                                           │
                                   nvstreamdemux
                                  ╱           ╲
                      (per stream)             ...
                   queue ──► nvdsosd ──► nvvideoconvert
                         ──► nvjpegenc ──► multifilesink

Buffer probes
─────────────
• Primary nvinfer src pad → extract detections, run AE gate, gate logic.
• nvinferserver src pad  → extract VLM text (if VLM enabled).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import structlog

from src import elements as E
from src import metadata as M
from src.constants import DSConstants, PipelineCfg, ModelCfg, PathCfg
from src.environment import get_gst
from src.rtsp_source import RTSPSource
from src.triton_config import generate_triton_config, normalize_model_version
from src.ae_gate import AutoencoderGate
from src.gate import DualGate
from src.metadata_writer import MetadataWriter

logger = structlog.get_logger()


class PipelineBuilder:
    """
    Builds and owns the GStreamer pipeline for VisionSentry.

    Parameters
    ----------
    stream_urls        : RTSP stream URLs (one per stream).
    inference_config   : Path to nvinfer .txt config for YOLO.
    output_dir         : Base directory for saved frames + metadata.
    gpu_id             : GPU to use for all DeepStream elements.
    width / height     : nvstreammux muxed frame dimensions.
    max_frames         : Stop saving after this many frames (None = unlimited).
    enable_vis         : Add nvdsosd bounding-box overlay.
    enable_frame_saving: Save JPEG frames to disk.
    vlm_grpc_endpoint  : Triton server host:port (None = VLM disabled).
    vlm_model_name     : Triton model name.
    vlm_model_version  : Triton model version (None = latest).
    vlm_input_tensor   : Tensor name for image input.
    vlm_output_tensor  : Tensor name for text output.
    vlm_infer_interval : Throttle probe: only pass every Nth frame with a person.
    """

    def __init__(
        self,
        stream_urls: List[str],
        inference_config: str,
        output_dir: str = PathCfg.DATA_FRAMES_DIR,
        gpu_id: int = PipelineCfg.DEFAULT_GPU_ID,
        width: int = PipelineCfg.DEFAULT_WIDTH,
        height: int = PipelineCfg.DEFAULT_HEIGHT,
        max_frames: Optional[int] = None,
        enable_vis: bool = True,
        enable_frame_saving: bool = True,
        vlm_grpc_endpoint: Optional[str] = None,
        vlm_model_name: Optional[str] = None,
        vlm_model_version: Optional[str] = None,
        vlm_input_tensor: str = "image_input",
        vlm_output_tensor: str = "text_output",
        vlm_infer_interval: int = 25,
    ) -> None:
        self.stream_urls = stream_urls
        self.inference_config = inference_config
        self.output_dir = Path(output_dir)
        self.gpu_id = gpu_id
        self.width = width
        self.height = height
        self.max_frames = max_frames
        self.enable_vis = enable_vis
        self.enable_frame_saving = enable_frame_saving

        # VLM
        self.vlm_endpoint = vlm_grpc_endpoint
        self.vlm_model_name = vlm_model_name
        self.vlm_model_version = normalize_model_version(vlm_model_version)
        self.vlm_input_tensor = vlm_input_tensor
        self.vlm_output_tensor = vlm_output_tensor
        self.vlm_infer_interval = vlm_infer_interval

        # Internals
        self.batch_size = min(len(stream_urls), 30)
        self.pipeline: Optional[Any] = None
        self._sources: List[RTSPSource] = []
        self._nvinfer: Optional[Any] = None
        self._nvinferserver: Optional[Any] = None
        self._converter: Optional[Any] = None
        self._demuxer: Optional[Any] = None

        # Gates + output helpers (set by orchestrator)
        self.ae_gate: Optional[AutoencoderGate] = None
        self.dual_gate: Optional[DualGate] = None
        self.pending_metadata: Dict[Tuple[int, int], Dict] = {}
        self.frames_with_metadata: set = set()
        self.vlm_outputs: Dict[Tuple[int, int], str] = {}
        self._metadata_writer: Optional[MetadataWriter] = None
        self._frames_saved = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Public: build + getters
    # ──────────────────────────────────────────────────────────────────────────

    def build(self) -> Any:
        """Assemble the complete GStreamer pipeline. Returns the Gst.Pipeline."""
        Gst = get_gst()
        self.pipeline = Gst.Pipeline()

        self._build_sources()
        self._build_muxer()
        self._connect_sources_to_muxer()
        self._build_nvinfer()

        if self.vlm_endpoint and self.vlm_model_name:
            self._build_nvinferserver()

        self._build_converter()

        if self.enable_frame_saving:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._metadata_writer = MetadataWriter(str(self.output_dir))
            self._build_demux_and_sinks()
        else:
            self._build_fakesink()

        self._setup_bus()
        logger.info("Pipeline built",
                    streams=len(self.stream_urls),
                    batch_size=self.batch_size,
                    vlm=bool(self.vlm_endpoint and self.vlm_model_name))
        return self.pipeline

    def get_nvinfer_src_pad(self) -> Optional[Any]:
        if self._nvinfer:
            return self._nvinfer.get_static_pad("src")
        return None

    def get_nvinferserver_src_pad(self) -> Optional[Any]:
        if self._nvinferserver:
            return self._nvinferserver.get_static_pad("src")
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Private: pipeline construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_sources(self) -> None:
        for idx, url in enumerate(self.stream_urls):
            src = RTSPSource(uri=url, stream_id=idx)
            if src.create_and_add(self.pipeline):
                self._sources.append(src)

    def _build_muxer(self) -> None:
        self._muxer = E.make_nvstreammux(
            name="nvstreammux",
            batch_size=self.batch_size,
            width=self.width,
            height=self.height,
            gpu_id=self.gpu_id,
            max_fps_n=PipelineCfg.MUXER_MAX_FPS_N,
            max_fps_d=PipelineCfg.MUXER_MAX_FPS_D,
            push_timeout_ns=PipelineCfg.MUXER_PUSH_TIMEOUT_NS,
        )
        self.pipeline.add(self._muxer)

    def _connect_sources_to_muxer(self) -> None:
        for src in self._sources:
            src.connect_to_muxer(self._muxer)

    def _build_nvinfer(self) -> None:
        self._nvinfer = E.make_nvinfer(
            name="primary-nvinfer",
            config_path=str(Path(self.inference_config).resolve()),
            batch_size=self.batch_size,
            gpu_id=self.gpu_id,
        )
        self.pipeline.add(self._nvinfer)
        if not self._muxer.link(self._nvinfer):
            raise RuntimeError("Failed to link nvstreammux → nvinfer")

    def _build_nvinferserver(self) -> None:
        config_path = generate_triton_config(
            grpc_endpoint=self.vlm_endpoint,
            model_name=self.vlm_model_name,
            model_version=self.vlm_model_version,
            batch_size=self.batch_size,
            input_tensor=self.vlm_input_tensor,
            output_tensor=self.vlm_output_tensor,
        )
        self._nvinferserver = E.make_nvinferserver(
            name="vlm-nvinferserver",
            config_path=config_path,
            gpu_id=self.gpu_id,
            infer_interval=self.vlm_infer_interval,
        )
        self.pipeline.add(self._nvinferserver)
        if not self._nvinfer.link(self._nvinferserver):
            raise RuntimeError("Failed to link nvinfer → nvinferserver")
        # Add VLM throttle probe (person + every Nth frame)
        self._add_vlm_throttle_probe(self._nvinferserver)

    def _build_converter(self) -> None:
        self._converter = E.make_nvvideoconvert("nvvideoconvert-main", self.gpu_id)
        self.pipeline.add(self._converter)
        prev = (self._nvinferserver or self._nvinfer)
        if not prev.link(self._converter):
            raise RuntimeError("Failed to link inference → nvvideoconvert")

    def _build_demux_and_sinks(self) -> None:
        """Build: nvstreamdemux → (per stream) queue → [nvdsosd] → nvvideoconvert → nvjpegenc → multifilesink."""
        Gst = get_gst()
        num = len(self._sources)

        self._demuxer = E.make_nvstreamdemux("nvstreamdemux")
        self.pipeline.add(self._demuxer)
        if not self._converter.link(self._demuxer):
            raise RuntimeError("Failed to link nvvideoconvert → nvstreamdemux")

        for sid in range(num):
            stream_dir = self.output_dir / f"stream_{sid}"
            stream_dir.mkdir(parents=True, exist_ok=True)

            # Request demux src pad
            demux_pad = self._demuxer.get_request_pad(f"src_{sid}")
            if not demux_pad:
                logger.warning("Could not allocate demux pad", stream_id=sid)
                continue

            # queue
            q = E.make_queue(f"queue-{sid}")
            self.pipeline.add(q)
            q_sink = q.get_static_pad("sink")
            if demux_pad.link(q_sink) != Gst.PadLinkReturn.OK:
                raise RuntimeError(f"Failed to link demux→queue for stream {sid}")

            prev_elem = q

            # optional nvdsosd
            if self.enable_vis:
                osd = E.make_nvdsosd(f"nvdsosd-{sid}", self.gpu_id)
                self.pipeline.add(osd)
                if not prev_elem.link(osd):
                    raise RuntimeError(f"Failed to link queue→nvdsosd stream {sid}")
                prev_elem = osd

            # nvvideoconvert (format conversion before jpegenc)
            conv = E.make_nvvideoconvert(f"conv-{sid}", self.gpu_id)
            self.pipeline.add(conv)
            if not prev_elem.link(conv):
                raise RuntimeError(f"Failed to link →nvvideoconvert stream {sid}")

            # nvjpegenc
            jpegenc = E.make_nvjpegenc(f"jpegenc-{sid}", quality=95)
            self.pipeline.add(jpegenc)
            if not conv.link(jpegenc):
                raise RuntimeError(f"Failed to link nvvideoconvert→nvjpegenc stream {sid}")

            # multifilesink
            location = str(stream_dir / "frame_%06d.jpg")
            sink = E.make_multifilesink(f"multifilesink-{sid}", location)
            self.pipeline.add(sink)
            if not jpegenc.link(sink):
                raise RuntimeError(f"Failed to link nvjpegenc→multifilesink stream {sid}")

            # Attach per-stream frame-tracking probes
            self._add_frame_tracking_probe(conv, jpegenc, sid, stream_dir)

            logger.info("Stream sink chain built", stream_id=sid,
                        output=str(stream_dir))

    def _build_fakesink(self) -> None:
        sink = E.make_fakesink("fakesink")
        self.pipeline.add(sink)
        prev = self._converter
        if not prev.link(sink):
            raise RuntimeError("Failed to link converter → fakesink")

    def _setup_bus(self) -> None:
        Gst = get_gst()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()

        def _on_message(bus: Any, msg: Any) -> None:
            mtype = msg.type
            if mtype == Gst.MessageType.EOS:
                logger.info("Pipeline EOS")
            elif mtype == Gst.MessageType.ERROR:
                err, dbg = msg.parse_error()
                logger.error("Pipeline error", error=str(err), debug=dbg)

        bus.connect("message", _on_message)

    # ──────────────────────────────────────────────────────────────────────────
    # Probes
    # ──────────────────────────────────────────────────────────────────────────

    def _add_vlm_throttle_probe(self, nvinferserver_elem: Any) -> None:
        """
        Drop buffers on the nvinferserver sink unless:
          - frame_number % vlm_infer_interval == 0
          - AND at least one person detected (class_id == 0)
        """
        Gst = get_gst()
        sink_pad = nvinferserver_elem.get_static_pad("sink")
        if not sink_pad:
            logger.warning("VLM throttle: cannot get nvinferserver sink pad")
            return

        interval = self.vlm_infer_interval

        def _throttle(pad: Any, info: Any, data: Any) -> Any:
            try:
                buf = info.get_buffer()
                if not buf:
                    return Gst.PadProbeReturn.OK
                batch_meta = M.get_batch_meta(buf)
                if not batch_meta:
                    return Gst.PadProbeReturn.OK

                import pyds  # type: ignore
                l_frame = batch_meta.frame_meta_list
                while l_frame is not None:
                    try:
                        fm = pyds.NvDsFrameMeta.cast(l_frame.data)
                        if fm.frame_num % interval == 0:
                            l_obj = fm.obj_meta_list
                            while l_obj is not None:
                                try:
                                    om = pyds.NvDsObjectMeta.cast(l_obj.data)
                                    if om and om.class_id == ModelCfg.PERSON_CLASS_ID:
                                        return Gst.PadProbeReturn.OK
                                except Exception:
                                    pass
                                finally:
                                    l_obj = l_obj.next
                    except Exception:
                        pass
                    finally:
                        l_frame = l_frame.next
                return Gst.PadProbeReturn.DROP
            except Exception:
                return Gst.PadProbeReturn.OK

        sink_pad.add_probe(Gst.PadProbeType.BUFFER, _throttle, None)
        logger.info("VLM throttle probe attached",
                    interval=interval,
                    note=f"VLM sees frames where frame_num%{interval}==0 AND person detected")

    def add_detection_probe(
        self,
        on_detection: Callable[[int, int, List[Dict], float], None],
    ) -> None:
        """
        Attach a buffer probe on the nvinfer src pad.

        Callback signature:
            on_detection(frame_number, stream_id, detections, max_yolo_conf) -> None
        """
        Gst = get_gst()
        pad = self.get_nvinfer_src_pad()
        if not pad:
            logger.warning("Cannot attach detection probe — nvinfer src pad not found")
            return

        def _probe(pad: Any, info: Any, data: Any) -> Any:
            try:
                buf = info.get_buffer()
                if not buf:
                    return Gst.PadProbeReturn.OK
                batch_meta = M.get_batch_meta(buf)
                if not batch_meta:
                    return Gst.PadProbeReturn.OK

                import pyds  # type: ignore
                l_frame = batch_meta.frame_meta_list
                while l_frame is not None:
                    try:
                        fm = pyds.NvDsFrameMeta.cast(l_frame.data)
                        frame_info = M.extract_frame_info(fm)
                        detections = M.extract_detections(fm)
                        max_conf = max(
                            (d["confidence"] for d in detections), default=0.0)
                        on_detection(
                            frame_info.get("frame_number", 0),
                            frame_info.get("stream_id", 0),
                            detections,
                            max_conf,
                        )
                    except Exception as exc:
                        logger.debug("Detection probe frame error", error=str(exc))
                    finally:
                        l_frame = l_frame.next
                return Gst.PadProbeReturn.OK
            except Exception as exc:
                logger.debug("Detection probe error", error=str(exc))
                return Gst.PadProbeReturn.OK

        pad.add_probe(Gst.PadProbeType.BUFFER, _probe, None)
        logger.info("Detection probe attached to nvinfer src pad")

    def add_vlm_output_probe(
        self,
        on_vlm: Callable[[int, int, str], None],
    ) -> None:
        """
        Attach a buffer probe on the nvinferserver src pad.
        Extracts VLM text and calls on_vlm(frame_number, stream_id, text).
        """
        Gst = get_gst()
        pad = self.get_nvinferserver_src_pad()
        if not pad:
            return

        output_tensor = self.vlm_output_tensor

        def _probe(pad: Any, info: Any, data: Any) -> Any:
            try:
                buf = info.get_buffer()
                if not buf:
                    return Gst.PadProbeReturn.OK
                batch_meta = M.get_batch_meta(buf)
                if not batch_meta:
                    return Gst.PadProbeReturn.OK

                import pyds  # type: ignore
                l_frame = batch_meta.frame_meta_list
                while l_frame is not None:
                    try:
                        fm = pyds.NvDsFrameMeta.cast(l_frame.data)
                        frame_info = M.extract_frame_info(fm)
                        text = M.extract_vlm_text(fm, tensor_name=output_tensor)
                        if text:
                            on_vlm(
                                frame_info.get("frame_number", 0),
                                frame_info.get("stream_id", 0),
                                text,
                            )
                    except Exception as exc:
                        logger.debug("VLM probe frame error", error=str(exc))
                    finally:
                        l_frame = l_frame.next
                return Gst.PadProbeReturn.OK
            except Exception as exc:
                logger.debug("VLM probe error", error=str(exc))
                return Gst.PadProbeReturn.OK

        pad.add_probe(Gst.PadProbeType.BUFFER, _probe, None)
        logger.info("VLM output probe attached to nvinferserver src pad")

    def _add_frame_tracking_probe(
        self,
        conv_elem: Any,
        jpegenc_elem: Any,
        stream_id: int,
        stream_dir: Path,
    ) -> None:
        """
        Two-probe approach for correct frame numbering:
          1. Probe on conv src pad  → extract frame_number, write metadata JSON.
          2. Probe on jpegenc src pad → save JPEG bytes with matching filename.
        """
        Gst = get_gst()
        frame_queue: deque = deque(maxlen=100)
        saved_frames: set = set()
        writer = self._metadata_writer
        pending = self.pending_metadata

        # ── Probe 1: metadata extraction (before jpegenc) ──────────────────
        conv_src = conv_elem.get_static_pad("src")
        if not conv_src:
            return

        def _meta_probe(pad: Any, info: Any, data: Any) -> Any:
            try:
                buf = info.get_buffer()
                if not buf:
                    return Gst.PadProbeReturn.OK
                batch_meta = M.get_batch_meta(buf)
                if not batch_meta:
                    return Gst.PadProbeReturn.OK

                import pyds  # type: ignore
                l_frame = batch_meta.frame_meta_list
                while l_frame is not None:
                    try:
                        fm = pyds.NvDsFrameMeta.cast(l_frame.data)
                        if fm.pad_index != stream_id:
                            l_frame = l_frame.next
                            continue

                        frame_num = fm.frame_num
                        key = (frame_num, stream_id)

                        # Look up prepared (gate-approved) metadata
                        meta = pending.get(key)
                        if meta and not meta.get("_saved", False):
                            detections = meta.get("detections", [])
                            if detections and writer:
                                writer.output_dir = stream_dir.parent
                                writer.write(
                                    meta["frame_info"],
                                    detections,
                                    vlm_output=meta.get("vlm_output"),
                                )
                                meta["_saved"] = True
                                frame_queue.append(frame_num)
                        l_frame = l_frame.next
                    except Exception:
                        l_frame = l_frame.next if l_frame else None
            except Exception:
                pass
            return Gst.PadProbeReturn.OK

        conv_src.add_probe(Gst.PadProbeType.BUFFER, _meta_probe, None)

        # ── Probe 2: JPEG save (after jpegenc) ─────────────────────────────
        jpegenc_src = jpegenc_elem.get_static_pad("src")
        if not jpegenc_src:
            return

        def _jpeg_probe(pad: Any, info: Any, data: Any) -> Any:
            try:
                buf = info.get_buffer()
                if not buf:
                    return Gst.PadProbeReturn.OK

                frame_num = frame_queue.popleft() if frame_queue else None
                if frame_num is None or frame_num in saved_frames:
                    return Gst.PadProbeReturn.OK

                out_path = stream_dir / f"frame_{frame_num:06d}.jpg"
                if not out_path.exists():
                    ok, map_info = buf.map(Gst.MapFlags.READ)
                    if ok:
                        try:
                            out_path.write_bytes(map_info.data)
                            saved_frames.add(frame_num)
                        finally:
                            buf.unmap(map_info)
            except Exception:
                pass
            return Gst.PadProbeReturn.OK

        jpegenc_src.add_probe(Gst.PadProbeType.BUFFER, _jpeg_probe, None)
