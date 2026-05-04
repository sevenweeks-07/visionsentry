"""
Metadata writer — saves frame_info + detections + VLM output to JSON.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger()


class MetadataWriter:
    """Writes per-frame JSON metadata files to a stream output directory."""

    def __init__(self, output_dir: str, enabled: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        frame_info: Dict[str, Any],
        detections: List[Dict[str, Any]],
        vlm_output: Optional[str] = None,
    ) -> Optional[Path]:
        """Write metadata JSON; returns path on success, None on failure."""
        if not self.enabled or not detections:
            return None

        frame_num = frame_info.get("frame_number", 0)
        stream_id = frame_info.get("stream_id", 0)
        stream_dir = self.output_dir / f"stream_{stream_id}"
        stream_dir.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "frame_info": frame_info,
            "detections": detections,
            "num_detections": len(detections),
        }
        if vlm_output:
            payload["vlm_output"] = vlm_output

        out_path = stream_dir / f"frame_{frame_num:06d}_info.json"
        try:
            out_path.write_text(json.dumps(payload, indent=2))
            return out_path
        except Exception as exc:
            logger.error("MetadataWriter.write failed", error=str(exc),
                         frame=frame_num, stream=stream_id)
            return None
