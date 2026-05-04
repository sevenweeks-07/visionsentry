"""
Dual-gate logic: YOLO (nvinfer) + Autoencoder.

Gate rule
---------
A frame passes if: yolo_flagged OR ae_flagged.
Frames that fail both gates are discarded — NOT sent to the VLM.

Every frame decision is logged to logs/gate_log.csv with columns:
  frame_id, stream_id, yolo_conf, ae_score, gate_passed, gate_reason

gate_reason is one of: "YOLO" | "AE" | "BOTH" | "SKIP"
"""

import csv
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import structlog

from src.constants import PathCfg

logger = structlog.get_logger()

_CSV_HEADER = ["frame_id", "stream_id", "yolo_conf", "ae_score",
               "gate_passed", "gate_reason"]


class GateLogger:
    """Appends gate decisions to a CSV file (one row per frame)."""

    def __init__(self, csv_path: str = PathCfg.GATE_LOG_CSV) -> None:
        self._path = Path(csv_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "w", newline="", buffering=1)
        self._writer = csv.DictWriter(self._file, fieldnames=_CSV_HEADER)
        self._writer.writeheader()

    def log(
        self,
        frame_id: int,
        stream_id: int,
        yolo_conf: float,
        ae_score: float,
        gate_passed: bool,
        gate_reason: str,
    ) -> None:
        self._writer.writerow({
            "frame_id": frame_id,
            "stream_id": stream_id,
            "yolo_conf": round(yolo_conf, 6),
            "ae_score": round(ae_score, 6),
            "gate_passed": gate_passed,
            "gate_reason": gate_reason,
        })

    def close(self) -> None:
        self._file.close()


def evaluate_gate(
    yolo_flagged: bool,
    ae_flagged: bool,
    yolo_conf: float,
    ae_score: float,
) -> Tuple[bool, str]:
    """
    Evaluate the dual-gate rule.

    Returns
    -------
    (gate_passed, gate_reason)
    """
    if yolo_flagged and ae_flagged:
        return True, "BOTH"
    if yolo_flagged:
        return True, "YOLO"
    if ae_flagged:
        return True, "AE"
    return False, "SKIP"


class DualGate:
    """
    High-level gate object used by the pipeline orchestrator.

    - Holds a GateLogger for CSV output.
    - Exposes decide() to evaluate one frame and log the result.
    """

    def __init__(self, csv_path: str = PathCfg.GATE_LOG_CSV) -> None:
        self._gate_logger = GateLogger(csv_path)
        self._total = 0
        self._passed = 0

    def decide(
        self,
        frame_id: int,
        stream_id: int,
        yolo_flagged: bool,
        yolo_conf: float,
        ae_flagged: bool,
        ae_score: float,
    ) -> Tuple[bool, str]:
        """
        Evaluate gate, log result, return (gate_passed, gate_reason).
        """
        gate_passed, gate_reason = evaluate_gate(
            yolo_flagged, ae_flagged, yolo_conf, ae_score)

        self._gate_logger.log(
            frame_id=frame_id,
            stream_id=stream_id,
            yolo_conf=yolo_conf,
            ae_score=ae_score,
            gate_passed=gate_passed,
            gate_reason=gate_reason,
        )

        self._total += 1
        if gate_passed:
            self._passed += 1

        if self._total % 100 == 0:
            pass_rate = self._passed / self._total * 100
            logger.info(
                "Gate stats",
                total=self._total,
                passed=self._passed,
                pass_rate_pct=round(pass_rate, 1),
            )

        return gate_passed, gate_reason

    def close(self) -> None:
        self._gate_logger.close()
