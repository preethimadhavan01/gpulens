"""
gpulens
=======
GPU workload analyzer for AI infrastructure teams on Kubernetes.

Detects performance and efficiency problems in GPU clusters:
  - GPU_IDLE_ALLOCATED      : GPU allocated to a pod but not computing
  - CPU_DATA_STARVATION     : Data pipeline starving GPU (CPU bottleneck)
  - MEMORY_PRESSURE         : GPU framebuffer near OOM
  - THERMAL_THROTTLING      : Temperature causing clock reduction
  - GPU_FRAGMENTATION       : Sparse GPU allocation wasting expensive nodes
  - TOPOLOGY_MISMATCH       : NVLink domain boundary crossing (PCIe fallback)
  - NCCL_STRAGGLER          : Slow node blocking distributed training collectives
  - COLD_START_LATENCY      : Pod scheduled but GPU not yet active

Quickstart
----------
>>> from gpulens.collectors import SyntheticCollector, Scenario
>>> from gpulens.analyzers import Analyzer
>>> from gpulens.reporters.cli_reporter import CLIReporter
>>>
>>> snapshot = SyntheticCollector(scenario=Scenario.MIXED).collect()
>>> report   = Analyzer().analyze(snapshot)
>>> CLIReporter().print_report(report)

CLI
---
  gpulens simulate --scenario mixed
  gpulens scan --prometheus-url http://prometheus.svc:9090
  gpulens scenarios
"""
__version__ = "0.1.0"

from gpulens.models import (
    ClusterSnapshot, NodeSnapshot, GPUMetrics, GPUType,
    Problem, ProblemSeverity, ProblemType, AnalysisReport,
)
from gpulens.analyzers import Analyzer
from gpulens.collectors import SyntheticCollector, PrometheusCollector, Scenario

__all__ = [
    "__version__",
    # Models
    "ClusterSnapshot", "NodeSnapshot", "GPUMetrics", "GPUType",
    "Problem", "ProblemSeverity", "ProblemType", "AnalysisReport",
    # Core API
    "Analyzer",
    # Collectors
    "SyntheticCollector", "PrometheusCollector", "Scenario",
]
