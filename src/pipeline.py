"""
VisionSentry Pipeline Builder.

Assembles a SIMPLE LINEAR GStreamer pipeline:
uridecodebin(s) ──► nvstreammux ──► nvinfer (YOLO) ──► nvinferserver (VLM) ──► nvvideoconvert ──► nvstreamdemux ──► sinks
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import structlog
import pyds 

from src import elements as E
from src import metadata as M
from src.constants import PipelineCfg, PathCfg
from src.environment import get_gst
from src.rtsp_source import RTSPSource
from src.triton_config import generate_triton_config, normalize_model_version

logger = structlog.get_logger()

class PipelineBuilder:
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
        self.vlm_endpoint = vlm_grpc_endpoint
        self.vlm_model_name = vlm_model_name
        self.vlm_model_version = normalize_model_version(vlm_model_version)
        self.vlm_input_tensor = vlm_input_tensor
        self.vlm_output_tensor = vlm_output_tensor
        self.vlm_infer_interval = vlm_infer_interval
        self.enable_frame_saving = enable_frame_saving

        self.batch_size = len(stream_urls)
        self.pipeline: Optional[Any] = None
        self._sources: List[RTSPSource] = []
        self._nvinfer: Optional[Any] = None
        self._nvinferserver: Optional[Any] = None
        self._converter: Optional[Any] = None
        self._demuxer: Optional[Any] = None
        
        self.pending_metadata: Dict[Tuple[int, int], Dict] = {}

    def build(self) -> Any:
        Gst = get_gst()
        self.pipeline = Gst.Pipeline()

        # 1. Sources
        for idx, url in enumerate(self.stream_urls):
            src = RTSPSource(uri=url, stream_id=idx)
            if src.create_and_add(self.pipeline):
                self._sources.append(src)

        # 2. Muxer
        self._muxer = E.make_nvstreammux("nvstreammux", self.batch_size, self.width, self.height, self.gpu_id)
        self._muxer.set_property("nvbuf-memory-type", 3) # Unified memory (Balanced)
        self.pipeline.add(self._muxer)
        for src in self._sources:
            src.connect_to_muxer(self._muxer)

        # 3. YOLO (pGIE-1)
        self._nvinfer = E.make_nvinfer("primary-nvinfer", str(Path(self.inference_config).resolve()), self.batch_size, self.gpu_id)
        self.pipeline.add(self._nvinfer)
        self._muxer.link(self._nvinfer)

        # 4. Autoencoder (pGIE-2)
        ae_config = Path("configs/autoencoder_infer.txt").resolve()
        self._ae_infer = E.make_nvinfer("ae-nvinfer", str(ae_config), self.batch_size, self.gpu_id)
        self.pipeline.add(self._ae_infer)
        self._nvinfer.link(self._ae_infer)
        last_elem = self._ae_infer

        # 5. VLM (Linear)
        if self.vlm_endpoint and self.vlm_model_name:
            config_path = generate_triton_config(self.vlm_endpoint, self.vlm_model_name, self.vlm_model_version, self.batch_size, self.vlm_input_tensor, self.vlm_output_tensor)
            self._nvinferserver = E.make_nvinferserver("vlm-nvinferserver", config_path, self.gpu_id, self.vlm_infer_interval)
            self.pipeline.add(self._nvinferserver)
            last_elem.link(self._nvinferserver)
            last_elem = self._nvinferserver

        # 5. OSD Drawing (GPU)
        self._osd_conv = E.make_nvvideoconvert("osd-conv", self.gpu_id)
        osd_caps = get_gst().Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
        self._osd_caps = E.make_capsfilter("osd-caps", osd_caps)
        self._osd = E.make_nvdsosd("nvdsosd-main", self.gpu_id)
        
        self.pipeline.add(self._osd_conv)
        self.pipeline.add(self._osd_caps)
        self.pipeline.add(self._osd)
        
        last_elem.link(self._osd_conv)
        self._osd_conv.link(self._osd_caps)
        self._osd_caps.link(self._osd)
        last_elem = self._osd

        # 6. Demuxer & Sinks
        self._demuxer = E.make_nvstreamdemux("nvstreamdemux")
        self.pipeline.add(self._demuxer)
        last_elem.link(self._demuxer)

        for sid in range(len(self._sources)):
            stream_dir = self.output_dir / f"stream_{sid}"
            stream_dir.mkdir(parents=True, exist_ok=True)
            
            demux_pad = self._demuxer.get_request_pad(f"src_{sid}")
            q = E.make_queue(f"queue-{sid}")
            conv = E.make_nvvideoconvert(f"conv-{sid}", self.gpu_id)
            jpegenc = E.make_nvjpegenc(f"jpegenc-{sid}")
            sink = E.make_multifilesink(f"multifilesink-{sid}", str(stream_dir / "frame_%06d.jpg"))
            
            self.pipeline.add(q); self.pipeline.add(conv); self.pipeline.add(jpegenc); self.pipeline.add(sink)
            demux_pad.link(q.get_static_pad("sink"))
            q.link(conv); conv.link(jpegenc); jpegenc.link(sink)

        return self.pipeline

    def add_detection_probe(self, callback: Callable) -> None:
        Gst = get_gst()
        # Use Autoencoder GIE src pad (GPU/NVMM)
        # Detections and AE scores are both available here in the metadata
        pad = self._ae_infer.get_static_pad("src")
        def _probe(pad, info, data):
            buf = info.get_buffer()
            if not buf: return Gst.PadProbeReturn.OK
            batch_meta = M.get_batch_meta(buf)
            if not batch_meta: return Gst.PadProbeReturn.OK
            l_frame = batch_meta.frame_meta_list
            while l_frame:
                fm = pyds.NvDsFrameMeta.cast(l_frame.data)
                detections = M.extract_detections(fm)
                max_conf = max((d["confidence"] for d in detections), default=0.0)
                callback(buf, fm, detections, max_conf)
                l_frame = l_frame.next
            return Gst.PadProbeReturn.OK
        pad.add_probe(Gst.PadProbeType.BUFFER, _probe, None)

    def add_vlm_output_probe(self, callback: Callable) -> None:
        if not self._nvinferserver: return
        Gst = get_gst()
        pad = self._nvinferserver.get_static_pad("src")
        def _probe(pad, info, data):
            buf = info.get_buffer()
            if not buf: return Gst.PadProbeReturn.OK
            batch_meta = M.get_batch_meta(buf)
            if not batch_meta: return Gst.PadProbeReturn.OK
            l_frame = batch_meta.frame_meta_list
            while l_frame:
                fm = pyds.NvDsFrameMeta.cast(l_frame.data)
                text = M.extract_vlm_text(fm, self.vlm_output_tensor)
                if text: callback(fm.frame_num, fm.pad_index, text)
                l_frame = l_frame.next
            return Gst.PadProbeReturn.OK
        pad.add_probe(Gst.PadProbeType.BUFFER, _probe, None)
