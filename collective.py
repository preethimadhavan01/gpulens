"""
gpulens.analyzers.collective
=============================
Detects NCCL_STRAGGLER: a node in a distributed training job whose GPU
utilization is significantly below the median of other nodes in the same job.

How NCCL all-reduce works
--------------------------
In data-parallel distributed training, every node computes gradients locally,
then participates in an all-reduce collective (ring-allreduce or tree-reduce)
to sum gradients across all ranks.

All-reduce is a *synchronous barrier*: no node can proceed to the next forward
pass until every node has contributed its gradients. If one node is 5x slower
(due to a failed NIC, degraded IB link, or hardware fault), the entire job
runs at 1/5th the optimal throughput.

Signals we use
--------------
1. Per-job GPU utilization median: the median across all nodes in the same job.
   A straggler node has GPU util << median (waiting at barrier, not computing).

2. InfiniBand receive bandwidth: a degraded IB link will show significantly
   lower rx_gbps compared to healthy nodes. This provides corroborating evidence
   for the root cause (NIC fault vs software issue).

3. Straggler ratio: node_util / median_util. Threshold default 0.5 means
   "this node is running at less than 50% of the job's median throughput."

Limitations
-----------
Without per-step timing data (requires NCCL hook or PyTorch profiler), we
infer the straggler from averaged utilization. This approach works well for
steady-state training but may miss transient stalls.
"""
from __future__ import annotations

from typing import Dict, List

from gpulens.models.cluster import ClusterSnapshot, GPUMetrics, NodeSnapshot
from gpulens.models.problems import Problem, ProblemSeverity, ProblemType


class CollectiveAnalyzer:
    """
    Detects straggler nodes in distributed training jobs.

    Parameters
    ----------
    straggler_ratio_threshold : float
        A node is flagged as straggler if its avg GPU util / job median GPU util
        falls below this ratio. Default 0.5 = below 50% of job median.
    min_job_nodes : int
        Minimum number of nodes in a job before straggler analysis runs.
        Single-node jobs cannot have stragglers.
    min_median_util : float
        Skip jobs whose median GPU util is below this threshold (job may not
        be in steady-state training — could be initialising or loading weights).
    """

    def __init__(
        self,
        straggler_ratio_threshold: float = 0.50,
        min_job_nodes: int               = 2,
        min_median_util: float           = 25.0,
    ) -> None:
        self.straggler_ratio_threshold = straggler_ratio_threshold
        self.min_job_nodes             = min_job_nodes
        self.min_median_util           = min_median_util

    def analyze(self, snapshot: ClusterSnapshot) -> List[Problem]:
        # Aggregate per-job node stats across the cluster
        job_stats: Dict[str, List[Dict]] = self._collect_job_stats(snapshot)
        problems: List[Problem] = []

        for job_name, node_stats in job_stats.items():
            if len(node_stats) < self.min_job_nodes:
                continue

            utils    = sorted(n["avg_gpu_util"] for n in node_stats)
            median_u = utils[len(utils) // 2]

            if median_u < self.min_median_util:
                continue  # whole job idle or starting — skip

            ib_rxs    = sorted(n["ib_rx_gbps"] for n in node_stats)
            median_ib = ib_rxs[len(ib_rxs) // 2]

            for node_data in node_stats:
                ratio = node_data["avg_gpu_util"] / median_u if median_u > 0 else 1.0
                if ratio >= self.straggler_ratio_threshold:
                    continue

                ib_rx      = node_data["ib_rx_gbps"]
                ib_degraded = median_ib > 10.0 and ib_rx < (median_ib * 0.3)

                problems.append(self._build_problem(
                    node_name   = node_data["node"],
                    job_name    = job_name,
                    node_data   = node_data,
                    ratio       = ratio,
                    median_util = median_u,
                    median_ib   = median_ib,
                    ib_degraded = ib_degraded,
                    total_nodes = len(node_stats),
                ))

        return problems

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _collect_job_stats(
        self, snapshot: ClusterSnapshot
    ) -> Dict[str, List[Dict]]:
        """
        Returns {job_name: [{node, avg_gpu_util, ib_rx_gbps, gpu_count}]}
        Only includes nodes that have ≥2 GPUs in the given job
        (single-GPU pods are not collective participants).
        """
        stats: Dict[str, List[Dict]] = {}

        for node in snapshot.nodes:
            job_gpus: Dict[str, List[GPUMetrics]] = {}
            for gpu in node.gpus:
                if gpu.job_name and gpu.is_allocated:
                    job_gpus.setdefault(gpu.job_name, []).append(gpu)

            for job, gpus in job_gpus.items():
                if len(gpus) < 2:
                    continue  # not a collective participant
                avg_util = sum(g.utilization_sm_pct for g in gpus) / len(gpus)
                stats.setdefault(job, []).append({
                    "node":         node.node_name,
                    "avg_gpu_util": avg_util,
                    "ib_rx_gbps":   node.infiniband_rx_gbps,
                    "ib_tx_gbps":   node.infiniband_tx_gbps,
                    "gpu_count":    len(gpus),
                })

        return stats

    def _build_problem(
        self,
        node_name:   str,
        job_name:    str,
        node_data:   Dict,
        ratio:       float,
        median_util: float,
        median_ib:   float,
        ib_degraded: bool,
        total_nodes: int,
    ) -> Problem:
        ib_rx    = node_data["ib_rx_gbps"]
        node_u   = node_data["avg_gpu_util"]
        gpu_cnt  = node_data["gpu_count"]
        eff_pct  = ratio * 100

        if ib_degraded:
            root_cause = (
                f"Root cause: InfiniBand link degraded — rx bandwidth {ib_rx:.1f} GB/s "
                f"vs cluster median {median_ib:.1f} GB/s. "
                "Likely causes: NIC hardware fault, cable issue, or IB switch port error."
            )
            primary_action = (
                "Run `ibstat` to check link state and `perfquery` for error counters. "
                "Check `ethtool -S <ib-interface>` for symbol errors and link flaps."
            )
        else:
            root_cause = (
                "Root cause unclear from available metrics. Possible: software hang, "
                "checkpointing overhead, or disk I/O blocking gradient computation."
            )
            primary_action = (
                "Attach NCCL_DEBUG=INFO and check logs for timeout or ring-election errors. "
                "Check pod for disk pressure (`kubectl describe pod`) and OOMKill events."
            )

        return Problem(
            type        = ProblemType.NCCL_STRAGGLER,
            severity    = ProblemSeverity.CRITICAL,
            node_name   = node_name,
            gpu_indices = list(range(gpu_cnt)),
            job_name    = job_name,
            description = (
                f"Node '{node_name}' is a straggler in distributed job '{job_name}'. "
                f"Average GPU utilization: {node_u:.1f}% "
                f"vs job median: {median_util:.1f}% ({eff_pct:.0f}% of expected). "
                f"{root_cause}"
            ),
            impact = (
                f"All-reduce collectives block all {total_nodes} nodes at gradient sync barriers. "
                f"Entire job throughput is capped at ~{eff_pct:.0f}% of optimal. "
                f"With {total_nodes} nodes participating, every training step is delayed until "
                f"this node completes its gradient computation. "
                "If this is a 7-day training run, expect an additional "
                f"~{(1/ratio - 1) * 100:.0f}% wall-clock time overhead."
            ),
            recommendation = (
                f"Immediate diagnosis:\n"
                f"  - {primary_action}\n"
                "  - Inspect DCGM for NvLink or PCIe error counters on this node.\n"
                "  - Check `dmesg` for hardware errors on the host.\n"
                "\n"
                "Mitigation while diagnosing:\n"
                "  - Use elastic training / fault-tolerant checkpointing "
                "    (PyTorch FSDP + torchrun --max_restarts) to resume if job crashes.\n"
                "  - Implement NCCL watchdog with timeout and automatic job restart.\n"
                "\n"
                "Long-term monitoring:\n"
                "  - Alert on per-node IB bandwidth drop below 150 GB/s for >2 minutes.\n"
                "  - Track GPU util variance across nodes in a job (max/min ratio > 2x = alert).\n"
                "  - Add DCGM NVML field DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL to Prometheus."
            ),
            metrics = {
                "node_avg_gpu_util_pct":    round(node_u, 1),
                "job_median_gpu_util_pct":  round(median_util, 1),
                "utilization_ratio":        round(ratio, 3),
                "effective_throughput_pct": round(eff_pct, 1),
                "ib_rx_gbps":               round(ib_rx, 2),
                "ib_tx_gbps":               round(node_data["ib_tx_gbps"], 2),
                "job_median_ib_rx_gbps":    round(median_ib, 2),
                "ib_degraded":              ib_degraded,
                "nodes_in_job":             total_nodes,
                "gpus_on_node":             gpu_cnt,
            },
        )
