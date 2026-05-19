"""
gpulens.analyzers.topology
==========================
Detects TOPOLOGY_MISMATCH: multi-GPU jobs assigned to GPUs that straddle
NVLink domain boundaries, forcing model-parallel communication over PCIe.

Background
----------
On 8-GPU A100/H100 SXM nodes, NVLink creates a high-bandwidth fabric:
  - A100 SXM4: 600 GB/s bidirectional between any pair of GPUs (full mesh)
  - H100 SXM5: 900 GB/s bidirectional (NVLink 4.0)

PCIe Gen4 x16 bandwidth:
  - ~32 GB/s bidirectional per slot
  - Cross-socket PCIe adds NUMA penalty (typically 50% further reduction)

For tensor-parallel and pipeline-parallel LLM inference/training, all-reduce
and point-to-point ops between GPUs in different NVLink domains will use PCIe,
degrading throughput by 10-30x vs NVLink.

A correctly topology-aware scheduler (using nvidia.com/gpu.topology labels)
will always assign a 4-GPU job to GPUs {0,1,2,3} or {4,5,6,7} — never
a split like {2,3,4,5} that crosses the boundary.

This is a silent failure: the job runs, NCCL selects a ring topology that
crosses PCIe, and throughput is reduced without any error or warning.

Reference topology
------------------
A100 SXM4 DGX A100 NVLink topology (simplified):
  Switch 0 (NVLink domain A): GPU 0, 1, 2, 3
  Switch 1 (NVLink domain B): GPU 4, 5, 6, 7
  Cross-switch links exist but at lower aggregate bandwidth.

Note: The actual topology is a full mesh, but cross-domain transfers
are routed through fewer NVLink hops, making them slower in practice.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Set

from gpulens.models.cluster import ClusterSnapshot, GPUMetrics, NodeSnapshot
from gpulens.models.problems import Problem, ProblemSeverity, ProblemType

# NVLink domain definitions for standard 8-GPU SXM nodes.
# Extend this dict for different GPU counts / form factors.
_NVLINK_DOMAINS: Dict[int, List[FrozenSet[int]]] = {
    8: [frozenset({0, 1, 2, 3}), frozenset({4, 5, 6, 7})],
    4: [frozenset({0, 1, 2, 3})],   # 4-GPU nodes: single domain
}

_NVLINK_BW_GBPS  = 600   # A100 SXM4 NVLink 3.0 bidirectional
_PCIE_BW_GBPS    = 32    # PCIe Gen4 x16 bidirectional
_BW_RATIO        = _NVLINK_BW_GBPS / _PCIE_BW_GBPS   # ~18.75x


class TopologyAnalyzer:
    """
    Identifies multi-GPU jobs that cross NVLink domain boundaries,
    forcing collective communication over PCIe.
    """

    def __init__(self, gpu_count: int = 8) -> None:
        self.domains = _NVLINK_DOMAINS.get(gpu_count, _NVLINK_DOMAINS[8])

    def analyze(self, snapshot: ClusterSnapshot) -> List[Problem]:
        problems: List[Problem] = []
        for node in snapshot.nodes:
            problems.extend(self._check_node(node))
        return problems

    def _check_node(self, node: NodeSnapshot) -> List[Problem]:
        # Group allocated GPUs by job name
        job_gpus: Dict[str, List[GPUMetrics]] = {}
        for gpu in node.gpus:
            if gpu.job_name and gpu.is_allocated:
                job_gpus.setdefault(gpu.job_name, []).append(gpu)

        problems: List[Problem] = []
        for job, gpus in job_gpus.items():
            if len(gpus) < 2:
                continue  # single-GPU job — topology irrelevant

            gpu_indices: Set[int] = {g.gpu_index for g in gpus}
            problem = self._check_placement(gpu_indices, job, node)
            if problem:
                problems.append(problem)

        return problems

    def _check_placement(
        self,
        gpu_indices: Set[int],
        job_name: str,
        node: NodeSnapshot,
    ) -> Problem | None:
        """
        Check if the given set of GPU indices fits within a single NVLink domain.
        If the job uses ≤ domain_size GPUs but straddles domains, flag it.
        """
        domain_size = max(len(d) for d in self.domains)

        # If job uses all GPUs on the node, it's using full node — topology OK
        if len(gpu_indices) >= node.gpu_count:
            return None

        # Check if all GPUs are within a single NVLink domain
        in_single_domain = any(gpu_indices.issubset(domain) for domain in self.domains)
        if in_single_domain:
            return None

        # Determine which domains are touched
        touched_domains = [
            sorted(domain)
            for domain in self.domains
            if domain & gpu_indices  # non-empty intersection
        ]

        # Estimate bandwidth impact
        cross_domain_pairs = sum(
            1
            for i in gpu_indices
            for j in gpu_indices
            if i < j and not any(i in d and j in d for d in self.domains)
        )
        total_pairs = len(gpu_indices) * (len(gpu_indices) - 1) // 2
        cross_pct   = (cross_domain_pairs / total_pairs * 100) if total_pairs > 0 else 0

        return Problem(
            type        = ProblemType.TOPOLOGY_MISMATCH,
            severity    = ProblemSeverity.HIGH,
            node_name   = node.node_name,
            gpu_indices = sorted(gpu_indices),
            job_name    = job_name,
            description = (
                f"Job '{job_name}' is using GPUs {sorted(gpu_indices)} "
                f"which span {len(touched_domains)} NVLink domain(s): {touched_domains}. "
                f"{cross_pct:.0f}% of GPU pairs must communicate over PCIe instead of NVLink."
            ),
            impact = (
                f"NVLink bandwidth: {_NVLINK_BW_GBPS} GB/s. "
                f"PCIe fallback: {_PCIE_BW_GBPS} GB/s — {_BW_RATIO:.0f}x slower. "
                "For tensor-parallel all-reduce at LLM scale, this can reduce "
                "effective throughput by 30-60% and increase step time proportionally. "
                "The slowdown is silent: the job runs but significantly underperforms."
            ),
            recommendation = (
                "Topology-aware scheduling:\n"
                "  - Install NVIDIA GPU Operator with topology manager policy 'restricted' "
                "    or 'best-effort' (kubelet --topology-manager-policy).\n"
                "  - Add `nvidia.com/gpu.topology: NVLink` node labels via DCGM Feature Discovery.\n"
                "  - Use `resources.limits['nvidia.com/gpu']` with pod affinity rules to "
                "    pin jobs to a single NVLink domain.\n"
                "Immediate workaround:\n"
                "  - Re-schedule the job specifying explicit GPU indices via CUDA_VISIBLE_DEVICES.\n"
                "  - Set NCCL_P2P_LEVEL=NVL to prevent NCCL from selecting cross-domain rings.\n"
                "Validation:\n"
                "  - Run `nvidia-smi topo -m` on the node to inspect NVLink connectivity matrix."
            ),
            metrics = {
                "job_gpu_indices":            sorted(gpu_indices),
                "nvlink_domains":             [sorted(d) for d in self.domains],
                "touched_domains":            touched_domains,
                "cross_domain_gpu_pairs":     cross_domain_pairs,
                "cross_domain_pairs_pct":     round(cross_pct, 1),
                "nvlink_bw_gbps":             _NVLINK_BW_GBPS,
                "pcie_bw_gbps":               _PCIE_BW_GBPS,
            },
        )
