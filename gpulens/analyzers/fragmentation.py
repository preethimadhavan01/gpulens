"""
gpulens.analyzers.fragmentation
================================
Detects GPU_FRAGMENTATION: nodes where a small number of GPUs are allocated,
making the remaining GPUs unschedulable while the node is billed as a full unit.

Why this matters for GPU cloud providers
-----------------------------------------
GPU nodes (DGX A100, HGX H100) are billed per-node-hour. A node with 2/8 GPUs
allocated is 75% wasted capacity — but Kubernetes sees it as "has available GPUs"
and won't necessarily consolidate or de-provision it.

Fragmentation compounds: many small inference deployments that each claim 1-2 GPUs
leave a cluster riddled with partially-occupied nodes that neither large training
jobs nor other small jobs can pack efficiently.
"""
from __future__ import annotations

from typing import List

from gpulens.models.cluster import ClusterSnapshot, NodeSnapshot
from gpulens.models.problems import Problem, ProblemSeverity, ProblemType

# Rough cost per GPU per hour (USD) used for impact estimates
_COST_PER_GPU_HOUR_USD = 3.0   # A100 80GB on-demand, approx 2024-2025


class FragmentationAnalyzer:
    """
    Detects nodes where GPU allocation is sparse, wasting capacity.

    Parameters
    ----------
    allocation_threshold : float
        Nodes with allocation ratio below this are flagged as fragmented.
        Default 0.5 = less than 50% of GPUs allocated.
    min_wasted_gpus : int
        Minimum number of unallocated GPUs before flagging (avoids noise
        on small-GPU nodes where 1 unallocated GPU is acceptable).
    """

    def __init__(
        self,
        allocation_threshold: float = 0.5,
        min_wasted_gpus: int = 2,
    ) -> None:
        self.allocation_threshold = allocation_threshold
        self.min_wasted_gpus      = min_wasted_gpus

    def analyze(self, snapshot: ClusterSnapshot) -> List[Problem]:
        cluster_frag_score = self._cluster_fragmentation(snapshot)
        problems: List[Problem] = []

        for node in snapshot.nodes:
            p = self._check_node(node, cluster_frag_score)
            if p:
                problems.append(p)

        return problems

    def _cluster_fragmentation(self, snapshot: ClusterSnapshot) -> float:
        """Cluster-wide fragmentation ratio (0 = fully packed, 1 = fully empty)."""
        total = snapshot.total_gpus
        if total == 0:
            return 0.0
        return 1.0 - (snapshot.allocated_gpus / total)

    def _check_node(
        self,
        node: NodeSnapshot,
        cluster_frag_score: float,
    ) -> Problem | None:
        n_total     = node.gpu_count
        n_allocated = len(node.allocated_gpus)
        n_wasted    = n_total - n_allocated

        if n_total == 0:
            return None

        alloc_ratio = n_allocated / n_total

        if alloc_ratio > self.allocation_threshold or n_wasted < self.min_wasted_gpus:
            return None  # not fragmented

        # Also check: how many allocated GPUs are themselves idle?
        idle_allocated = len(node.idle_allocated_gpus)

        severity = (
            ProblemSeverity.HIGH
            if alloc_ratio < 0.25
            else ProblemSeverity.MEDIUM
        )

        hourly_waste = n_wasted * _COST_PER_GPU_HOUR_USD
        idle_note    = (
            f" Additionally, {idle_allocated} allocated GPU(s) are themselves idle."
            if idle_allocated > 0
            else ""
        )

        return Problem(
            type        = ProblemType.GPU_FRAGMENTATION,
            severity    = severity,
            node_name   = node.node_name,
            gpu_indices = [g.gpu_index for g in node.unallocated_gpus],
            description = (
                f"Node has {n_allocated}/{n_total} GPUs allocated "
                f"({alloc_ratio * 100:.0f}% utilization of node capacity). "
                f"{n_wasted} GPUs are idle and blocking larger jobs from scheduling.{idle_note}"
            ),
            impact = (
                f"Node is billed as a full {n_total}-GPU unit but only {alloc_ratio * 100:.0f}% "
                f"of capacity is in use. Estimated waste: ~${hourly_waste:.0f}/hr "
                f"(${hourly_waste * 24:.0f}/day) at current GPU pricing. "
                f"Cluster-wide fragmentation score: {cluster_frag_score * 100:.0f}%."
            ),
            recommendation = (
                "Node consolidation:\n"
                "  - Enable Karpenter node consolidation or Cluster Autoscaler with "
                "    `--scale-down-enabled=true` and `--scale-down-utilization-threshold=0.5`.\n"
                "  - Use pod disruption budgets to allow safe eviction of fragmented nodes.\n"
                "Scheduling policy:\n"
                "  - Switch scheduler to `LeastAllocated` → `MostAllocated` bin-packing.\n"
                "  - Apply `TopologySpreadConstraints` to pack inference pods tightly.\n"
                "Workload design:\n"
                "  - For low-QPS inference, use NVIDIA Triton with multiple models per GPU.\n"
                "  - Consider MIG (Multi-Instance GPU) to serve multiple small models per physical GPU.\n"
                "  - Set GPU resource requests as fractional (0.25, 0.5) for small inference workloads."
            ),
            metrics = {
                "allocated_gpus":             n_allocated,
                "total_gpus":                 n_total,
                "unallocated_gpus":           n_wasted,
                "allocation_ratio":           round(alloc_ratio, 3),
                "idle_allocated_gpus":        idle_allocated,
                "cluster_fragmentation_pct":  round(cluster_frag_score * 100, 1),
                "estimated_waste_per_hour_usd": round(hourly_waste, 2),
            },
        )
