"""
gpulens.models.cluster
======================
Core data structures representing a point-in-time snapshot of a GPU cluster.

Designed to be populated from:
  - DCGM Prometheus exporter (live)
  - kube-state-metrics (pod attribution)
  - SyntheticCollector (offline / demo)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


class GPUType(str, Enum):
    A100_80G = "NVIDIA A100 80GB SXM4"
    A100_40G = "NVIDIA A100 40GB PCIe"
    H100_80G = "NVIDIA H100 80GB SXM5"
    H100_NVL = "NVIDIA H100 NVL"
    A10G     = "NVIDIA A10G"
    V100_32G = "NVIDIA V100 32GB SXM2"
    UNKNOWN  = "Unknown"

    @property
    def memory_mib(self) -> float:
        return {
            GPUType.A100_80G: 81920.0,
            GPUType.A100_40G: 40960.0,
            GPUType.H100_80G: 81920.0,
            GPUType.H100_NVL: 94208.0,
            GPUType.A10G:     24576.0,
            GPUType.V100_32G: 32768.0,
            GPUType.UNKNOWN:  40960.0,
        }[self]

    @property
    def tdp_watts(self) -> float:
        return {
            GPUType.A100_80G: 400.0,
            GPUType.A100_40G: 300.0,
            GPUType.H100_80G: 700.0,
            GPUType.H100_NVL: 400.0,
            GPUType.A10G:     150.0,
            GPUType.V100_32G: 300.0,
            GPUType.UNKNOWN:  300.0,
        }[self]


@dataclass
class PodEvent:
    """Kubernetes pod lifecycle event relevant to GPU readiness."""
    pod_name: str
    namespace: str
    event_type: str   # "Scheduled" | "ContainersReady" | "Running" | "FirstGPUKernel"
    timestamp: datetime


@dataclass
class GPUMetrics:
    """
    Per-GPU metrics at a single point in time.
    Maps directly to DCGM Prometheus metric labels.
    """
    gpu_index: int
    uuid: str

    # Compute
    utilization_sm_pct: float        # DCGM_FI_DEV_GPU_UTIL    — SM utilization 0-100
    utilization_memory_pct: float    # DCGM_FI_DEV_MEM_COPY_UTIL — memory BW util 0-100

    # Memory
    memory_used_mib: float           # DCGM_FI_DEV_FB_USED
    memory_total_mib: float          # DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE

    # Clocks
    sm_clock_mhz: float              # DCGM_FI_DEV_SM_CLOCK
    memory_clock_mhz: float          # DCGM_FI_DEV_MEM_CLOCK

    # Thermal / power
    power_watts: float               # DCGM_FI_DEV_POWER_USAGE
    temperature_c: float             # DCGM_FI_DEV_GPU_TEMP

    # Kubernetes attribution (from kube-state-metrics)
    pod_name: Optional[str] = None
    namespace: Optional[str] = None
    job_name: Optional[str] = None          # batch.kubernetes.io/job-name label
    container_name: Optional[str] = None

    # Topology
    nvlink_connected_gpus: List[int] = field(default_factory=list)
    is_mig_enabled: bool = False
    mig_profile: Optional[str] = None       # e.g. "3g.40gb"

    @property
    def memory_utilization_pct(self) -> float:
        if self.memory_total_mib == 0:
            return 0.0
        return (self.memory_used_mib / self.memory_total_mib) * 100.0

    @property
    def is_allocated(self) -> bool:
        return self.pod_name is not None

    @property
    def is_idle(self) -> bool:
        return self.is_allocated and self.utilization_sm_pct < 10.0

    @property
    def memory_free_mib(self) -> float:
        return self.memory_total_mib - self.memory_used_mib


@dataclass
class NodeSnapshot:
    """
    Aggregated metrics for a single GPU node at a point in time.
    One NodeSnapshot contains one GPUMetrics per physical GPU.
    """
    node_name: str
    gpu_count: int
    gpu_type: GPUType
    gpus: List[GPUMetrics]

    # Host system metrics
    cpu_utilization_pct: float
    memory_utilization_pct: float
    network_rx_gbps: float
    network_tx_gbps: float
    infiniband_rx_gbps: float        # Key signal for NCCL collective health
    infiniband_tx_gbps: float

    pod_events: List[PodEvent] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # ── Derived properties ─────────────────────────────────────────────────

    @property
    def allocated_gpus(self) -> List[GPUMetrics]:
        return [g for g in self.gpus if g.is_allocated]

    @property
    def unallocated_gpus(self) -> List[GPUMetrics]:
        return [g for g in self.gpus if not g.is_allocated]

    @property
    def idle_allocated_gpus(self) -> List[GPUMetrics]:
        return [g for g in self.gpus if g.is_idle]

    @property
    def avg_gpu_utilization(self) -> float:
        if not self.gpus:
            return 0.0
        return sum(g.utilization_sm_pct for g in self.gpus) / len(self.gpus)

    @property
    def avg_gpu_memory_utilization(self) -> float:
        if not self.gpus:
            return 0.0
        return sum(g.memory_utilization_pct for g in self.gpus) / len(self.gpus)

    @property
    def allocation_ratio(self) -> float:
        if self.gpu_count == 0:
            return 0.0
        return len(self.allocated_gpus) / self.gpu_count


@dataclass
class ClusterSnapshot:
    """
    Full cluster state at a point in time.
    The primary input to all gpulens analyzers.
    """
    cluster_name: str
    nodes: List[NodeSnapshot]
    snapshot_time: datetime = field(default_factory=datetime.utcnow)
    analysis_window_minutes: int = 15

    # ── Derived properties ─────────────────────────────────────────────────

    @property
    def total_gpus(self) -> int:
        return sum(n.gpu_count for n in self.nodes)

    @property
    def allocated_gpus(self) -> int:
        return sum(len(n.allocated_gpus) for n in self.nodes)

    @property
    def unallocated_gpus(self) -> int:
        return self.total_gpus - self.allocated_gpus

    @property
    def all_gpus(self) -> List[GPUMetrics]:
        return [g for n in self.nodes for g in n.gpus]

    @property
    def cluster_avg_utilization(self) -> float:
        gpus = self.all_gpus
        if not gpus:
            return 0.0
        return sum(g.utilization_sm_pct for g in gpus) / len(gpus)

    @property
    def allocation_ratio(self) -> float:
        if self.total_gpus == 0:
            return 0.0
        return self.allocated_gpus / self.total_gpus

    def node_by_name(self, name: str) -> Optional[NodeSnapshot]:
        return next((n for n in self.nodes if n.node_name == name), None)
