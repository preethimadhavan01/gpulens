"""
gpulens.models.problems
=======================
Problem taxonomy and AnalysisReport.

Each Problem represents one semantically meaningful issue detected by an Analyzer.
Problems are self-contained: they include what happened, what it costs, and how to fix it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from gpulens.models.cluster import ClusterSnapshot


class ProblemSeverity(str, Enum):
    """
    Severity definitions follow SRE on-call convention:
      CRITICAL  — Ongoing revenue/SLA impact; investigate immediately
      HIGH      — Significant waste or performance degradation; fix within hours
      MEDIUM    — Noteworthy inefficiency; fix within the week
      LOW       — Optimization opportunity; address in next planning cycle
      INFO      — Informational; no action required
    """
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"
    INFO     = "INFO"

    @property
    def rich_style(self) -> str:
        return {
            "CRITICAL": "bold red",
            "HIGH":     "bold orange1",
            "MEDIUM":   "bold yellow",
            "LOW":      "bold blue",
            "INFO":     "dim",
        }[self.value]

    @property
    def icon(self) -> str:
        return {
            "CRITICAL": "🔴",
            "HIGH":     "🟠",
            "MEDIUM":   "🟡",
            "LOW":      "🔵",
            "INFO":     "⚪",
        }[self.value]

    @property
    def sort_order(self) -> int:
        return {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}[self.value]


class ProblemType(str, Enum):
    """
    Enumerated problem types, each with a stable identifier string.
    Stable identifiers allow downstream consumers (alerting, ticketing) to key on them.
    """
    # Utilization problems
    GPU_IDLE_ALLOCATED   = "GPU_IDLE_ALLOCATED"    # GPU allocated, near-zero compute
    CPU_DATA_STARVATION  = "CPU_DATA_STARVATION"   # GPU starved by data pipeline
    MEMORY_PRESSURE      = "MEMORY_PRESSURE"       # GPU memory near capacity
    THERMAL_THROTTLING   = "THERMAL_THROTTLING"    # Temp-induced clock reduction

    # Structural / scheduling problems
    GPU_FRAGMENTATION    = "GPU_FRAGMENTATION"     # Sparse allocation on expensive nodes
    TOPOLOGY_MISMATCH    = "TOPOLOGY_MISMATCH"     # NVLink domain boundary crossing

    # Distributed training problems
    NCCL_STRAGGLER       = "NCCL_STRAGGLER"        # Slow node blocking collective ops
    COLD_START_LATENCY   = "COLD_START_LATENCY"    # Pod scheduled but GPU inactive


@dataclass
class Problem:
    """
    A single detected performance or efficiency problem.

    Intended to be actionable: every problem carries its own impact estimate
    and concrete remediation steps so on-call engineers don't need to context-switch.
    """
    type: ProblemType
    severity: ProblemSeverity
    node_name: str
    description: str     # What is happening, with observed values
    impact: str          # Business/performance cost
    recommendation: str  # Concrete steps to fix

    metrics: Dict[str, Any] = field(default_factory=dict)    # Raw supporting numbers
    gpu_indices: List[int]  = field(default_factory=list)
    pod_name: Optional[str] = None
    job_name: Optional[str] = None
    detected_at: datetime   = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type":           self.type.value,
            "severity":       self.severity.value,
            "node":           self.node_name,
            "pod":            self.pod_name,
            "job":            self.job_name,
            "gpu_indices":    self.gpu_indices,
            "description":    self.description,
            "impact":         self.impact,
            "recommendation": self.recommendation,
            "metrics":        self.metrics,
            "detected_at":    self.detected_at.isoformat(),
        }


@dataclass
class AnalysisReport:
    """
    The output of running all analyzers against a ClusterSnapshot.
    Returned by Analyzer.analyze() and consumed by reporters.
    """
    cluster_snapshot: ClusterSnapshot
    problems: List[Problem] = field(default_factory=list)
    analysis_duration_ms: float = 0.0

    # ── Derived views ──────────────────────────────────────────────────────

    @property
    def critical_problems(self) -> List[Problem]:
        return [p for p in self.problems if p.severity == ProblemSeverity.CRITICAL]

    @property
    def has_critical(self) -> bool:
        return bool(self.critical_problems)

    @property
    def by_severity(self) -> Dict[str, List[Problem]]:
        result: Dict[str, List[Problem]] = {s.value: [] for s in ProblemSeverity}
        for p in self.problems:
            result[p.severity.value].append(p)
        return result

    @property
    def by_type(self) -> Dict[str, List[Problem]]:
        result: Dict[str, List[Problem]] = {}
        for p in self.problems:
            result.setdefault(p.type.value, []).append(p)
        return result

    @property
    def by_node(self) -> Dict[str, List[Problem]]:
        result: Dict[str, List[Problem]] = {}
        for p in self.problems:
            result.setdefault(p.node_name, []).append(p)
        return result

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        snap = self.cluster_snapshot
        return {
            "cluster":                  snap.cluster_name,
            "snapshot_time":            snap.snapshot_time.isoformat(),
            "analysis_window_minutes":  snap.analysis_window_minutes,
            "cluster_stats": {
                "total_gpus":          snap.total_gpus,
                "allocated_gpus":      snap.allocated_gpus,
                "allocation_ratio":    round(snap.allocation_ratio, 3),
                "avg_utilization_pct": round(snap.cluster_avg_utilization, 1),
                "node_count":          len(snap.nodes),
            },
            "problem_count": len(self.problems),
            "problems_by_severity": {
                k: len(v) for k, v in self.by_severity.items()
            },
            "problems": [p.to_dict() for p in self.problems],
            "analysis_duration_ms": round(self.analysis_duration_ms, 2),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
