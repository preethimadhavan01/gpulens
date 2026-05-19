"""
gpulens.analyzers.utilization
==============================
Detects utilization-class problems on individual GPUs.

Problems detected
-----------------
  GPU_IDLE_ALLOCATED   — GPU allocated to a pod but SM utilization near zero.
  CPU_DATA_STARVATION  — GPU starved by data pipeline (low GPU util + pinned CPU).
  MEMORY_PRESSURE      — GPU framebuffer near capacity; XLA/CUDA degradation likely.
  THERMAL_THROTTLING   — GPU temperature above safe threshold; clocks being reduced.
"""
from __future__ import annotations

from typing import List

from gpulens.models.cluster import ClusterSnapshot, GPUMetrics, NodeSnapshot
from gpulens.models.problems import Problem, ProblemSeverity, ProblemType


class UtilizationAnalyzer:
    """
    Per-GPU utilization analysis.

    Parameters
    ----------
    idle_threshold : float
        SM utilization % below which an allocated GPU is considered idle.
    memory_pressure_threshold : float
        GPU framebuffer % above which memory pressure is flagged.
    cpu_starvation_threshold : float
        Host CPU % above which a data pipeline bottleneck is suspected.
    thermal_threshold_c : float
        Temperature °C above which thermal throttling risk is flagged.
    """

    def __init__(
        self,
        idle_threshold: float            = 10.0,
        memory_pressure_threshold: float = 90.0,
        cpu_starvation_threshold: float  = 90.0,
        thermal_threshold_c: float       = 83.0,
    ) -> None:
        self.idle_threshold            = idle_threshold
        self.memory_pressure_threshold = memory_pressure_threshold
        self.cpu_starvation_threshold  = cpu_starvation_threshold
        self.thermal_threshold_c       = thermal_threshold_c

    def analyze(self, snapshot: ClusterSnapshot) -> List[Problem]:
        problems: List[Problem] = []
        for node in snapshot.nodes:
            problems.extend(self._check_node(node))
        return problems

    # ── Node-level dispatch ─────────────────────────────────────────────────

    def _check_node(self, node: NodeSnapshot) -> List[Problem]:
        problems: List[Problem] = []
        cpu_pinned = node.cpu_utilization_pct >= self.cpu_starvation_threshold

        for gpu in node.gpus:
            if not gpu.is_allocated:
                continue  # unallocated — skip; FragmentationAnalyzer handles those

            if gpu.utilization_sm_pct < self.idle_threshold:
                if cpu_pinned:
                    problems.append(self._cpu_starvation(gpu, node))
                else:
                    problems.append(self._idle_allocated(gpu, node))
            elif cpu_pinned and gpu.utilization_sm_pct < 60.0:
                # GPU above the strict "idle" threshold but CPU is saturated —
                # classic DataLoader starvation: GPU bursts high then stalls
                # waiting for the next batch. Average util lands 20-55%, high variance.
                problems.append(self._cpu_starvation(gpu, node))

            if gpu.memory_utilization_pct >= self.memory_pressure_threshold:
                problems.append(self._memory_pressure(gpu, node))

            if gpu.temperature_c >= self.thermal_threshold_c:
                problems.append(self._thermal(gpu, node))

        return problems

    # ── Problem constructors ────────────────────────────────────────────────

    def _idle_allocated(self, gpu: GPUMetrics, node: NodeSnapshot) -> Problem:
        severity = (
            ProblemSeverity.HIGH
            if gpu.utilization_sm_pct < 2.0
            else ProblemSeverity.MEDIUM
        )
        return Problem(
            type        = ProblemType.GPU_IDLE_ALLOCATED,
            severity    = severity,
            node_name   = node.node_name,
            gpu_indices = [gpu.gpu_index],
            pod_name    = gpu.pod_name,
            job_name    = gpu.job_name,
            description = (
                f"GPU {gpu.gpu_index} is allocated to pod '{gpu.pod_name}' "
                f"but SM utilization is only {gpu.utilization_sm_pct:.1f}%. "
                f"Memory footprint: {gpu.memory_used_mib:.0f}/{gpu.memory_total_mib:.0f} MiB "
                f"({gpu.memory_utilization_pct:.1f}%)."
            ),
            impact = (
                "Allocated GPU compute is effectively wasted — blocking other workloads "
                "from scheduling while delivering no throughput. "
                "At typical cloud pricing (~$3/GPU/hr for A100), this wastes compute budget."
            ),
            recommendation = (
                "1. Check if pod is in init phase or blocked on data ingestion.\n"
                "2. If serving inference at low QPS, enable GPU sharing via MIG "
                "   (multi-instance GPU) or NVIDIA time-slicing.\n"
                "3. Set Kubernetes resource quotas and eviction policies to reclaim "
                "   idle GPUs after a defined SLO window.\n"
                "4. Add liveness probe that checks for GPU activity; restart idle pods."
            ),
            metrics = {
                "gpu_util_sm_pct":     gpu.utilization_sm_pct,
                "gpu_util_mem_pct":    gpu.utilization_memory_pct,
                "memory_used_mib":     gpu.memory_used_mib,
                "memory_total_mib":    gpu.memory_total_mib,
                "power_watts":         gpu.power_watts,
                "temperature_c":       gpu.temperature_c,
            },
        )

    def _cpu_starvation(self, gpu: GPUMetrics, node: NodeSnapshot) -> Problem:
        waste_pct = 100.0 - gpu.utilization_sm_pct
        return Problem(
            type        = ProblemType.CPU_DATA_STARVATION,
            severity    = ProblemSeverity.HIGH,
            node_name   = node.node_name,
            gpu_indices = [gpu.gpu_index],
            pod_name    = gpu.pod_name,
            job_name    = gpu.job_name,
            description = (
                f"GPU {gpu.gpu_index} SM utilization is {gpu.utilization_sm_pct:.1f}% "
                f"while host CPU is saturated at {node.cpu_utilization_pct:.1f}%. "
                "This pattern indicates the data loading pipeline cannot keep the GPU fed. "
                "The GPU is idle waiting for the next batch."
            ),
            impact = (
                f"GPU compute wasted {waste_pct:.0f}% of the time. "
                "With data-parallel training, all GPUs in the job suffer proportionally. "
                "A CPU bottleneck that halves GPU utilization effectively doubles your "
                "training cost for the same wall-clock time."
            ),
            recommendation = (
                "Immediate (no code change):\n"
                "  - Increase `num_workers` in DataLoader to match CPU core count.\n"
                "  - Enable `persistent_workers=True` to avoid fork overhead.\n"
                "  - Set `pin_memory=True` for faster host→GPU transfers.\n"
                "Longer term:\n"
                "  - Profile with `torch.utils.bottleneck` or NVIDIA Nsight to confirm.\n"
                "  - Migrate to NVIDIA DALI for GPU-accelerated data loading.\n"
                "  - Use WebDataset / MosaicML StreamingDataset for streaming from object store.\n"
                "  - Pre-tokenize and cache datasets to avoid per-epoch CPU preprocessing."
            ),
            metrics = {
                "gpu_util_sm_pct":  gpu.utilization_sm_pct,
                "cpu_util_pct":     node.cpu_utilization_pct,
                "memory_used_mib":  gpu.memory_used_mib,
                "power_watts":      gpu.power_watts,
            },
        )

    def _memory_pressure(self, gpu: GPUMetrics, node: NodeSnapshot) -> Problem:
        mem_pct  = gpu.memory_utilization_pct
        severity = (
            ProblemSeverity.CRITICAL
            if mem_pct >= 97.0
            else ProblemSeverity.HIGH
        )
        return Problem(
            type        = ProblemType.MEMORY_PRESSURE,
            severity    = severity,
            node_name   = node.node_name,
            gpu_indices = [gpu.gpu_index],
            pod_name    = gpu.pod_name,
            job_name    = gpu.job_name,
            description = (
                f"GPU {gpu.gpu_index} framebuffer at {mem_pct:.1f}% capacity "
                f"({gpu.memory_used_mib:.0f} / {gpu.memory_total_mib:.0f} MiB used). "
                "Near-OOM conditions trigger CUDA memory manager fallback paths, "
                "XLA graph recompilation, and activation recomputation."
            ),
            impact = (
                f"Estimated throughput degradation: 20-50% vs unconstrained run. "
                f"SM utilization has dropped to {gpu.utilization_sm_pct:.0f}% "
                "due to memory management overhead. "
                "OOM crash risk is high — a single spike will kill the training job."
            ),
            recommendation = (
                "Immediate:\n"
                "  - Reduce batch size by 25% and verify stability.\n"
                "  - Enable mixed precision (bf16/fp16): `model.to(torch.bfloat16)`.\n"
                "Throughput recovery:\n"
                "  - Enable gradient checkpointing: `model.gradient_checkpointing_enable()`.\n"
                "  - Use ZeRO Stage 2/3 (DeepSpeed) to shard optimizer states across GPUs.\n"
                "  - Enable activation offloading to CPU for transformer layers.\n"
                "Diagnosis:\n"
                "  - Run `torch.cuda.memory_summary()` to identify largest allocations.\n"
                "  - Check for memory leaks: validate that eval loop clears grad cache."
            ),
            metrics = {
                "memory_used_mib":  gpu.memory_used_mib,
                "memory_total_mib": gpu.memory_total_mib,
                "memory_pct":       round(mem_pct, 1),
                "gpu_util_sm_pct":  gpu.utilization_sm_pct,
                "temperature_c":    gpu.temperature_c,
            },
        )

    def _thermal(self, gpu: GPUMetrics, node: NodeSnapshot) -> Problem:
        return Problem(
            type        = ProblemType.THERMAL_THROTTLING,
            severity    = ProblemSeverity.MEDIUM,
            node_name   = node.node_name,
            gpu_indices = [gpu.gpu_index],
            pod_name    = gpu.pod_name,
            description = (
                f"GPU {gpu.gpu_index} temperature is {gpu.temperature_c:.0f}°C, "
                f"approaching thermal throttle threshold (typically 83°C for A100, 83°C for H100). "
                f"Current SM clock: {gpu.sm_clock_mhz:.0f} MHz, "
                f"power draw: {gpu.power_watts:.0f} W."
            ),
            impact = (
                "When temperature exceeds the throttle threshold, the GPU autonomously "
                "reduces SM and memory clocks to protect hardware integrity. "
                "This causes non-deterministic throughput degradation of 10-30%."
            ),
            recommendation = (
                "Infrastructure:\n"
                "  - Verify datacenter airflow and inlet temperature (target ≤ 25°C).\n"
                "  - Check that server fans are not blocked and rotating at rated RPM.\n"
                "  - Validate thermal interface material on heat spreader (degraded over time).\n"
                "Software mitigation:\n"
                "  - Power-cap GPUs below TDP to reduce heat: "
                "    `nvidia-smi -pl 350` (A100 80GB, down from 400W).\n"
                "  - Use DCGM health check to flag thermally stressed GPUs automatically."
            ),
            metrics = {
                "temperature_c":      gpu.temperature_c,
                "sm_clock_mhz":       gpu.sm_clock_mhz,
                "power_watts":        gpu.power_watts,
                "throttle_threshold": self.thermal_threshold_c,
            },
        )
