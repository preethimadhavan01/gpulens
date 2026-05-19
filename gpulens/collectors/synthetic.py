"""
gpulens.collectors.synthetic
============================
Generates realistic ClusterSnapshot objects with known problems embedded.
Used for offline development, demos, CI testing, and analyzer validation.

Each scenario deliberately injects specific problem patterns so you can
verify that the right analyzers fire on the right nodes.

Scenario → Expected problems:
  healthy          → none
  fragmentation    → GPU_FRAGMENTATION on nodes 0-3
  cpu_starvation   → CPU_DATA_STARVATION cluster-wide
  nccl_straggler   → NCCL_STRAGGLER on node-03
  cold_start       → GPU_IDLE_ALLOCATED on nodes 0-3 (new pods, no CUDA yet)
  memory_pressure  → MEMORY_PRESSURE cluster-wide
  mixed            → fragmentation(0-1), cpu_starvation(2), nccl_straggler(3),
                     memory_pressure(4-5), healthy(6-7)
"""
from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

from gpulens.collectors.base import BaseCollector
from gpulens.models.cluster import (
    ClusterSnapshot,
    GPUMetrics,
    GPUType,
    NodeSnapshot,
    PodEvent,
)


class Scenario(str, Enum):
    HEALTHY          = "healthy"
    GPU_FRAGMENTATION = "fragmentation"
    CPU_STARVATION   = "cpu_starvation"
    NCCL_STRAGGLER   = "nccl_straggler"
    COLD_START       = "cold_start"
    MEMORY_PRESSURE  = "memory_pressure"
    MIXED            = "mixed"


# A100 SXM4 8-GPU NVLink topology (simplified DGX A100 switch fabric):
# Full mesh — all 8 GPUs connected at 600 GB/s bidirectional.
# For cross-domain scheduling detection we split into two logical NVLink groups.
_NVLINK_GROUPS_8GPU = [frozenset({0, 1, 2, 3}), frozenset({4, 5, 6, 7})]


def _nvlink_peers(gpu_index: int, total_gpus: int = 8) -> List[int]:
    """Return the NVLink-connected peer GPU indices for a given GPU."""
    if total_gpus == 8:
        for group in _NVLINK_GROUPS_8GPU:
            if gpu_index in group:
                return sorted(group - {gpu_index})
    return []


class SyntheticCollector(BaseCollector):
    """
    Offline collector that produces deterministic, annotated cluster snapshots.

    Parameters
    ----------
    scenario : Scenario
        Which problem pattern to embed.
    node_count : int
        Number of GPU nodes to simulate (capped at 8 for "mixed" scenario).
    gpus_per_node : int
        GPUs per node (8 is typical for DGX/HGX class hardware).
    gpu_type : GPUType
        GPU model; affects memory capacity, TDP, and benchmark util ceilings.
    cluster_name : str
        Label embedded in the returned ClusterSnapshot.
    seed : Optional[int]
        Fix the random seed for reproducible output. None = non-deterministic.
    """

    def __init__(
        self,
        scenario: Scenario = Scenario.MIXED,
        node_count: int = 8,
        gpus_per_node: int = 8,
        gpu_type: GPUType = GPUType.A100_80G,
        cluster_name: str = "synthetic-cluster",
        seed: Optional[int] = 42,
    ) -> None:
        self.scenario      = scenario
        self.node_count    = node_count
        self.gpus_per_node = gpus_per_node
        self.gpu_type      = gpu_type
        self.cluster_name  = cluster_name
        if seed is not None:
            random.seed(seed)

    def is_available(self) -> bool:
        return True

    def collect(self) -> ClusterSnapshot:
        builders = {
            Scenario.HEALTHY:           self._healthy,
            Scenario.GPU_FRAGMENTATION: self._fragmentation,
            Scenario.CPU_STARVATION:    self._cpu_starvation,
            Scenario.NCCL_STRAGGLER:    self._nccl_straggler,
            Scenario.COLD_START:        self._cold_start,
            Scenario.MEMORY_PRESSURE:   self._memory_pressure,
            Scenario.MIXED:             self._mixed,
        }
        return builders[self.scenario]()

    # ── GPU / node factory helpers ──────────────────────────────────────────

    def _gpu(
        self,
        idx: int,
        util_sm: float      = 0.0,
        util_mem: float     = 0.0,
        mem_frac: float     = 0.0,
        power_frac: float   = 0.05,
        temp: float         = 42.0,
        pod_name: Optional[str]   = None,
        namespace: Optional[str]  = None,
        job_name: Optional[str]   = None,
        nvlink_peers: Optional[List[int]] = None,
    ) -> GPUMetrics:
        total_mib = self.gpu_type.memory_mib
        max_power = self.gpu_type.tdp_watts

        # Realistic measurement noise (~1-2% stdev)
        util_sm  = max(0.0, min(100.0, util_sm  + random.gauss(0, 1.5)))
        util_mem = max(0.0, min(100.0, util_mem + random.gauss(0, 1.0)))

        # SM clock scales with utilization (simplified model)
        sm_clock_base = 1410.0  # A100 base boost clock MHz
        sm_clock = sm_clock_base * (0.65 + 0.35 * util_sm / 100.0)

        return GPUMetrics(
            gpu_index             = idx,
            uuid                  = f"GPU-{uuid.uuid4().hex[:8].upper()}",
            utilization_sm_pct    = round(util_sm, 1),
            utilization_memory_pct= round(util_mem, 1),
            memory_used_mib       = round(total_mib * mem_frac),
            memory_total_mib      = total_mib,
            sm_clock_mhz          = round(sm_clock),
            memory_clock_mhz      = 1593.0,
            power_watts           = round(max_power * power_frac, 1),
            temperature_c         = round(temp + random.gauss(0, 0.5), 1),
            pod_name              = pod_name,
            namespace             = namespace,
            job_name              = job_name,
            nvlink_connected_gpus = nvlink_peers if nvlink_peers is not None
                                    else _nvlink_peers(idx, self.gpus_per_node),
        )

    def _node(
        self,
        name: str,
        gpus: List[GPUMetrics],
        cpu_util:  float = 30.0,
        mem_util:  float = 40.0,
        ib_rx_gbps: float = 0.0,
        ib_tx_gbps: float = 0.0,
        pod_events: Optional[List[PodEvent]] = None,
    ) -> NodeSnapshot:
        return NodeSnapshot(
            node_name             = name,
            gpu_count             = self.gpus_per_node,
            gpu_type              = self.gpu_type,
            gpus                  = gpus,
            cpu_utilization_pct   = round(max(0.0, min(100.0, cpu_util + random.gauss(0, 2.0))), 1),
            memory_utilization_pct= round(max(0.0, min(100.0, mem_util + random.gauss(0, 1.0))), 1),
            network_rx_gbps       = round(random.uniform(1, 5), 2),
            network_tx_gbps       = round(random.uniform(1, 5), 2),
            infiniband_rx_gbps    = round(max(0.0, ib_rx_gbps + random.gauss(0, 2.0)), 2),
            infiniband_tx_gbps    = round(max(0.0, ib_tx_gbps + random.gauss(0, 2.0)), 2),
            pod_events            = pod_events or [],
            timestamp             = datetime.utcnow(),
        )

    # ── Scenario builders ───────────────────────────────────────────────────

    def _healthy(self) -> ClusterSnapshot:
        """
        Baseline: all nodes running LLM training at high GPU utilization.
        Expected problems: NONE.
        """
        nodes = []
        for ni in range(self.node_count):
            job = f"llm-training-{ni // 2:02d}"
            pod = f"trainer-{ni:02d}"
            gpus = [
                self._gpu(
                    i, util_sm=random.uniform(88, 96), util_mem=random.uniform(72, 85),
                    mem_frac=0.78, power_frac=0.88, temp=random.uniform(72, 78),
                    pod_name=pod, namespace="ml-team",
                    job_name=job,
                )
                for i in range(self.gpus_per_node)
            ]
            nodes.append(self._node(f"gpu-node-{ni:02d}", gpus, cpu_util=45, mem_util=55,
                                     ib_rx_gbps=175, ib_tx_gbps=175))
        return ClusterSnapshot(self.cluster_name, nodes)

    def _fragmentation(self) -> ClusterSnapshot:
        """
        Half the cluster has 2/8 GPUs allocated to idle inference pods.
        The other half is running dense training.

        Expected problems: GPU_FRAGMENTATION on nodes 0 to node_count//2-1
        """
        nodes = []
        half = self.node_count // 2
        for ni in range(self.node_count):
            gpus = []
            if ni < half:
                # FRAGMENTED: 2 allocated (idle), 6 bare/unallocated
                for i in range(self.gpus_per_node):
                    if i < 2:
                        gpus.append(self._gpu(
                            i, util_sm=7.0, util_mem=4.0, mem_frac=0.12,
                            power_frac=0.11, temp=50.0,
                            pod_name=f"inf-server-{ni:02d}-{i}", namespace="inference",
                        ))
                    else:
                        gpus.append(self._gpu(i, temp=40.0))  # unallocated
            else:
                # HEALTHY: full 8-GPU training job
                job = f"llm-train-{ni:02d}"
                pod = f"trainer-{ni:02d}"
                for i in range(self.gpus_per_node):
                    gpus.append(self._gpu(
                        i, util_sm=91, util_mem=79, mem_frac=0.80,
                        power_frac=0.90, temp=75.0,
                        pod_name=pod, namespace="ml-team",
                        job_name=job,
                    ))
            nodes.append(self._node(f"gpu-node-{ni:02d}", gpus,
                                     cpu_util=12 if ni < half else 50,
                                     mem_util=25 if ni < half else 60,
                                     ib_rx_gbps=5 if ni < half else 170))
        return ClusterSnapshot(self.cluster_name, nodes)

    def _cpu_starvation(self) -> ClusterSnapshot:
        """
        Training job across all nodes. CPU pegged at 97% due to too few DataLoader
        workers. GPUs average ~35% SM utilization with high variance (bursts then stalls).

        Expected problems: CPU_DATA_STARVATION on all nodes.
        """
        nodes = []
        for ni in range(self.node_count):
            job = "resnet-train-bottlenecked"
            pod = f"trainer-{ni:02d}"
            gpus = [
                self._gpu(
                    i,
                    util_sm=random.uniform(22, 48),   # spiky average — high variance
                    util_mem=random.uniform(28, 52),
                    mem_frac=0.58, power_frac=0.42, temp=61.0,
                    pod_name=pod, namespace="ml-team",
                    job_name=job,
                )
                for i in range(self.gpus_per_node)
            ]
            nodes.append(self._node(f"gpu-node-{ni:02d}", gpus,
                                     cpu_util=97.0, mem_util=84.0,
                                     ib_rx_gbps=18, ib_tx_gbps=18))
        return ClusterSnapshot(self.cluster_name, nodes)

    def _nccl_straggler(self) -> ClusterSnapshot:
        """
        8-node all-reduce training job. Node 3 has a degraded IB link (NIC failure),
        falling back to Ethernet (~10 Gbps vs 200 Gbps InfiniBand).
        Node 3's GPUs are waiting on NCCL collectives → low util.
        All other nodes are blocked at barrier waiting for node 3.

        Expected problems: NCCL_STRAGGLER on gpu-node-03.
        """
        nodes = []
        job = "gpt4-pretrain-8node"
        straggler_node = 3

        for ni in range(self.node_count):
            is_straggler = (ni == straggler_node)
            gpu_util = random.uniform(12, 22) if is_straggler else random.uniform(72, 85)
            ib_bw    = random.uniform(0.8, 2.5) if is_straggler else random.uniform(148, 190)
            pod = f"trainer-{ni:02d}"

            gpus = [
                self._gpu(
                    i, util_sm=gpu_util, util_mem=gpu_util * 0.85,
                    mem_frac=0.72, power_frac=0.28 if is_straggler else 0.82,
                    temp=52.0 if is_straggler else random.uniform(71, 78),
                    pod_name=pod, namespace="ml-team",
                    job_name=job,
                )
                for i in range(self.gpus_per_node)
            ]
            nodes.append(self._node(
                f"gpu-node-{ni:02d}", gpus,
                cpu_util=22 if is_straggler else 52,
                ib_rx_gbps=ib_bw, ib_tx_gbps=ib_bw,
            ))
        return ClusterSnapshot(self.cluster_name, nodes)

    def _cold_start(self) -> ClusterSnapshot:
        """
        4 nodes have pods that Kubernetes says "Running" but GPU utilization is zero.
        Large container images + model weight downloads mean CUDA hasn't initialised.
        Pod events show pods were scheduled 8-14 minutes ago.

        Expected problems: GPU_IDLE_ALLOCATED on nodes 0-3.
        """
        nodes = []
        now = datetime.utcnow()
        for ni in range(self.node_count):
            cold = (ni < 4)
            if cold:
                gpus = [
                    self._gpu(
                        i, util_sm=0.0, util_mem=0.0, mem_frac=0.0,
                        power_frac=0.04, temp=42.0,
                        pod_name=f"inference-{ni:02d}-gpu{i}", namespace="inference",
                    )
                    for i in range(self.gpus_per_node)
                ]
                scheduled_ago = timedelta(minutes=random.uniform(8, 14))
                events = [
                    PodEvent(
                        pod_name  = f"inference-{ni:02d}-gpu{i}",
                        namespace = "inference",
                        event_type= "Scheduled",
                        timestamp = now - scheduled_ago,
                    )
                    for i in range(self.gpus_per_node)
                ]
                node = self._node(f"gpu-node-{ni:02d}", gpus, cpu_util=18,
                                   ib_rx_gbps=1, ib_tx_gbps=1, pod_events=events)
            else:
                job = f"stable-inference-{ni:02d}"
                gpus = [
                    self._gpu(
                        i, util_sm=random.uniform(58, 75), util_mem=random.uniform(52, 68),
                        mem_frac=0.62, power_frac=0.72, temp=69.0,
                        pod_name=f"inference-{ni:02d}-gpu{i}", namespace="inference",
                        job_name=job,
                    )
                    for i in range(self.gpus_per_node)
                ]
                node = self._node(f"gpu-node-{ni:02d}", gpus, cpu_util=32,
                                   ib_rx_gbps=60, ib_tx_gbps=60)
            nodes.append(node)
        return ClusterSnapshot(self.cluster_name, nodes)

    def _memory_pressure(self) -> ClusterSnapshot:
        """
        Training with too-large batch size. All GPUs near OOM.
        XLA/CUDA is recompiling graphs, causing degraded utilization.

        Expected problems: MEMORY_PRESSURE on all nodes.
        """
        nodes = []
        for ni in range(self.node_count):
            job = f"large-model-train-{ni:02d}"
            pod = f"trainer-{ni:02d}"
            gpus = [
                self._gpu(
                    i,
                    util_sm=random.uniform(38, 60),      # degraded due to recompile stalls
                    util_mem=random.uniform(88, 98),
                    mem_frac=random.uniform(0.92, 0.99),
                    power_frac=0.65, temp=random.uniform(78, 85),
                    pod_name=pod, namespace="ml-team",
                    job_name=job,
                )
                for i in range(self.gpus_per_node)
            ]
            nodes.append(self._node(f"gpu-node-{ni:02d}", gpus,
                                     cpu_util=52, mem_util=82,
                                     ib_rx_gbps=120, ib_tx_gbps=120))
        return ClusterSnapshot(self.cluster_name, nodes)

    def _mixed(self) -> ClusterSnapshot:
        """
        Realistic production cluster with overlapping problems across nodes.
        Hard-coded to 8 nodes for determinism (node_count is ignored for mixed).

        Node layout:
          00-01 → GPU_FRAGMENTATION  (2/8 GPUs, idle inference pods)
          02    → CPU_DATA_STARVATION (training, CPU pegged at 97%)
          03    → NCCL_STRAGGLER      (IB link degraded to ~1 Gbps)
          04-05 → MEMORY_PRESSURE     (near-OOM training)
          06-07 → Healthy             (baseline, 90%+ GPU util)
        """
        # Nodes 3, 6, 7 share job "dist-train-8node". Nodes 6+7 are healthy
        # peers at ~90% GPU util with full IB. Node 3 is the straggler (IB degraded).
        node_configs: Dict[int, dict] = {
            0: dict(kind="frag",       job=None),
            1: dict(kind="frag",       job=None),
            2: dict(kind="cpu_starved", job="batch-train-stalled"),
            3: dict(kind="straggler",  job="dist-train-8node"),
            4: dict(kind="mem_press",  job="large-model-00"),
            5: dict(kind="mem_press",  job="large-model-01"),
            6: dict(kind="healthy",    job="dist-train-8node"),
            7: dict(kind="healthy",    job="dist-train-8node"),
        }

        nodes = []
        for ni in range(8):
            cfg  = node_configs[ni]
            kind = cfg["kind"]
            job  = cfg["job"]
            name = f"gpu-node-{ni:02d}"

            if kind == "frag":
                gpus = []
                for i in range(self.gpus_per_node):
                    if i < 2:
                        gpus.append(self._gpu(i, util_sm=6.0, util_mem=3.5, mem_frac=0.10,
                                              power_frac=0.10, temp=50.0,
                                              pod_name=f"inf-{ni:02d}-gpu{i}",
                                              namespace="inference"))
                    else:
                        gpus.append(self._gpu(i, temp=40.0))
                nodes.append(self._node(name, gpus, cpu_util=11, mem_util=22,
                                         ib_rx_gbps=3, ib_tx_gbps=3))

            elif kind == "cpu_starved":
                pod = f"trainer-{ni:02d}"
                gpus = [
                    self._gpu(i, util_sm=random.uniform(20, 44), util_mem=random.uniform(24, 46),
                              mem_frac=0.57, power_frac=0.40, temp=60.0,
                              pod_name=pod, namespace="ml-team",
                              job_name=job)
                    for i in range(self.gpus_per_node)
                ]
                nodes.append(self._node(name, gpus, cpu_util=97, mem_util=86,
                                         ib_rx_gbps=14, ib_tx_gbps=14))

            elif kind == "straggler":
                pod = f"trainer-{ni:02d}"
                gpus = [
                    self._gpu(i, util_sm=random.uniform(11, 21), util_mem=random.uniform(9, 17),
                              mem_frac=0.70, power_frac=0.24, temp=53.0,
                              pod_name=pod, namespace="ml-team",
                              job_name=job)
                    for i in range(self.gpus_per_node)
                ]
                nodes.append(self._node(name, gpus, cpu_util=20,
                                         ib_rx_gbps=1.1, ib_tx_gbps=1.1))

            elif kind == "mem_press":
                pod = f"trainer-{ni:02d}"
                gpus = [
                    self._gpu(i, util_sm=random.uniform(36, 56), util_mem=random.uniform(89, 98),
                              mem_frac=random.uniform(0.92, 0.98), power_frac=0.63, temp=82.0,
                              pod_name=pod, namespace="ml-team",
                              job_name=job)
                    for i in range(self.gpus_per_node)
                ]
                nodes.append(self._node(name, gpus, cpu_util=51, mem_util=81,
                                         ib_rx_gbps=110, ib_tx_gbps=110))

            else:  # healthy
                pod = f"trainer-{ni:02d}"
                gpus = [
                    self._gpu(i, util_sm=random.uniform(88, 95), util_mem=random.uniform(73, 85),
                              mem_frac=0.78, power_frac=0.88, temp=74.0,
                              pod_name=pod, namespace="ml-team",
                              job_name=job)
                    for i in range(self.gpus_per_node)
                ]
                nodes.append(self._node(name, gpus, cpu_util=45, mem_util=58,
                                         ib_rx_gbps=172, ib_tx_gbps=172))

        return ClusterSnapshot(self.cluster_name, nodes)
