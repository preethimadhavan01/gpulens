"""
gpulens.analyzers
=================
Orchestrates all sub-analyzers and returns a unified AnalysisReport.

Usage
-----
>>> from gpulens.analyzers import Analyzer
>>> from gpulens.collectors import SyntheticCollector, Scenario
>>>
>>> snapshot = SyntheticCollector(scenario=Scenario.MIXED).collect()
>>> report   = Analyzer().analyze(snapshot)
>>> print(f"{len(report.problems)} problems found")
"""
from __future__ import annotations

import time
from typing import List

from gpulens.analyzers.collective import CollectiveAnalyzer
from gpulens.analyzers.fragmentation import FragmentationAnalyzer
from gpulens.analyzers.topology import TopologyAnalyzer
from gpulens.analyzers.utilization import UtilizationAnalyzer
from gpulens.models.cluster import ClusterSnapshot
from gpulens.models.problems import AnalysisReport, Problem


class Analyzer:
    """
    Runs all gpulens analyzers and returns an AnalysisReport.

    Parameters
    ----------
    idle_gpu_util_threshold : float
        GPU SM util % below which an allocated GPU is considered idle.
    memory_pressure_threshold : float
        GPU framebuffer % above which memory pressure is flagged.
    cpu_starvation_threshold : float
        Host CPU % above which data-pipeline starvation is suspected.
    straggler_ratio_threshold : float
        Ratio of node GPU util / job median below which node is a straggler.
    thermal_threshold_c : float
        GPU temperature °C above which throttling risk is flagged.
    """

    def __init__(
        self,
        idle_gpu_util_threshold:    float = 10.0,
        memory_pressure_threshold:  float = 90.0,
        cpu_starvation_threshold:   float = 90.0,
        straggler_ratio_threshold:  float = 0.50,
        thermal_threshold_c:        float = 83.0,
    ) -> None:
        self._sub_analyzers = [
            UtilizationAnalyzer(
                idle_threshold            = idle_gpu_util_threshold,
                memory_pressure_threshold = memory_pressure_threshold,
                cpu_starvation_threshold  = cpu_starvation_threshold,
                thermal_threshold_c       = thermal_threshold_c,
            ),
            FragmentationAnalyzer(),
            TopologyAnalyzer(),
            CollectiveAnalyzer(straggler_ratio_threshold=straggler_ratio_threshold),
        ]

    def analyze(self, snapshot: ClusterSnapshot) -> AnalysisReport:
        """
        Run all analyzers against the snapshot and return a ranked AnalysisReport.
        Problems are sorted by severity (CRITICAL first).
        """
        t0: float = time.monotonic()
        problems: List[Problem] = []

        for sub in self._sub_analyzers:
            problems.extend(sub.analyze(snapshot))

        problems.sort(key=lambda p: p.severity.sort_order)

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        return AnalysisReport(
            cluster_snapshot     = snapshot,
            problems             = problems,
            analysis_duration_ms = elapsed_ms,
        )


__all__ = ["Analyzer", "UtilizationAnalyzer", "FragmentationAnalyzer",
           "TopologyAnalyzer", "CollectiveAnalyzer"]
