"""
gpulens test suite
==================
Tests each analyzer with hand-crafted inputs where we know exactly what
problems should be detected. If these pass, the analyzers are correct.

Run with:
    pytest tests/ -v
    pytest tests/ -v --tb=short   # shorter tracebacks
"""
import pytest
from datetime import datetime

from gpulens.models.cluster import ClusterSnapshot, NodeSnapshot, GPUMetrics, GPUType
from gpulens.models.problems import ProblemType, ProblemSeverity
from gpulens.analyzers import Analyzer
from gpulens.analyzers.utilization import UtilizationAnalyzer
from gpulens.analyzers.fragmentation import FragmentationAnalyzer
from gpulens.analyzers.collective import CollectiveAnalyzer
from gpulens.analyzers.topology import TopologyAnalyzer
from gpulens.collectors.synthetic import SyntheticCollector, Scenario


# ── Helpers ─────────────────────────────────────────────────────────────────

def make_gpu(
    idx=0, util_sm=90.0, util_mem=80.0,
    mem_used_mib=65536, mem_total_mib=81920,
    power=300.0, temp=72.0,
    pod_name="trainer-0", namespace="ml-team",
    job_name="train-job", nvlink_peers=None,
) -> GPUMetrics:
    return GPUMetrics(
        gpu_index=idx,
        uuid=f"GPU-TEST-{idx:02d}",
        utilization_sm_pct=util_sm,
        utilization_memory_pct=util_mem,
        memory_used_mib=mem_used_mib,
        memory_total_mib=mem_total_mib,
        sm_clock_mhz=1410.0,
        memory_clock_mhz=1593.0,
        power_watts=power,
        temperature_c=temp,
        pod_name=pod_name,
        namespace=namespace,
        job_name=job_name,
        nvlink_connected_gpus=nvlink_peers or [],
    )


def make_node(
    name="gpu-node-00", gpus=None,
    cpu_util=30.0, mem_util=40.0,
    ib_rx=175.0, ib_tx=175.0,
) -> NodeSnapshot:
    if gpus is None:
        gpus = [make_gpu(i) for i in range(8)]
    return NodeSnapshot(
        node_name=name,
        gpu_count=len(gpus),
        gpu_type=GPUType.A100_80G,
        gpus=gpus,
        cpu_utilization_pct=cpu_util,
        memory_utilization_pct=mem_util,
        network_rx_gbps=5.0,
        network_tx_gbps=5.0,
        infiniband_rx_gbps=ib_rx,
        infiniband_tx_gbps=ib_tx,
        timestamp=datetime.utcnow(),
    )


def make_cluster(nodes, name="test-cluster") -> ClusterSnapshot:
    return ClusterSnapshot(cluster_name=name, nodes=nodes)


# ── Utilization Analyzer ─────────────────────────────────────────────────────

class TestUtilizationAnalyzer:

    def test_healthy_gpu_no_problems(self):
        """A GPU running at 90% util should generate zero problems."""
        node = make_node(gpus=[make_gpu(0, util_sm=91.0, temp=72.0)])
        snapshot = make_cluster([node])
        problems = UtilizationAnalyzer().analyze(snapshot)
        assert len(problems) == 0, f"Expected 0 problems, got: {[p.type for p in problems]}"

    def test_idle_allocated_gpu(self):
        """Allocated GPU at 5% SM util → GPU_IDLE_ALLOCATED."""
        gpu = make_gpu(0, util_sm=5.0, pod_name="my-pod")
        node = make_node(gpus=[gpu], cpu_util=20.0)
        problems = UtilizationAnalyzer().analyze(make_cluster([node]))

        idle = [p for p in problems if p.type == ProblemType.GPU_IDLE_ALLOCATED]
        assert len(idle) == 1
        assert idle[0].node_name == "gpu-node-00"
        assert idle[0].gpu_indices == [0]

    def test_unallocated_gpu_not_flagged(self):
        """An unallocated GPU (no pod) should never be flagged as idle."""
        gpu = make_gpu(0, util_sm=0.0, pod_name=None)
        node = make_node(gpus=[gpu])
        problems = UtilizationAnalyzer().analyze(make_cluster([node]))
        idle = [p for p in problems if p.type == ProblemType.GPU_IDLE_ALLOCATED]
        assert len(idle) == 0

    def test_cpu_starvation_high_cpu_low_gpu(self):
        """GPU at 35% util + CPU at 97% → CPU_DATA_STARVATION, not GPU_IDLE."""
        gpu = make_gpu(0, util_sm=35.0, pod_name="trainer-0")
        node = make_node(gpus=[gpu], cpu_util=97.0)
        problems = UtilizationAnalyzer().analyze(make_cluster([node]))

        starvation = [p for p in problems if p.type == ProblemType.CPU_DATA_STARVATION]
        idle       = [p for p in problems if p.type == ProblemType.GPU_IDLE_ALLOCATED]

        assert len(starvation) >= 1, "Expected CPU_DATA_STARVATION"
        assert len(idle) == 0, "Should not flag GPU_IDLE_ALLOCATED when CPU is the bottleneck"

    def test_cpu_starvation_also_fires_below_idle_threshold(self):
        """GPU at 5% util + CPU at 97% → CPU_DATA_STARVATION (not just GPU_IDLE)."""
        gpu = make_gpu(0, util_sm=5.0, pod_name="trainer-0")
        node = make_node(gpus=[gpu], cpu_util=97.0)
        problems = UtilizationAnalyzer().analyze(make_cluster([node]))

        types = [p.type for p in problems]
        assert ProblemType.CPU_DATA_STARVATION in types

    def test_memory_pressure_critical(self):
        """GPU framebuffer at 98% → MEMORY_PRESSURE CRITICAL."""
        total = 81920.0
        gpu = make_gpu(0, mem_used_mib=total * 0.98, mem_total_mib=total)
        node = make_node(gpus=[gpu])
        problems = UtilizationAnalyzer().analyze(make_cluster([node]))

        mem = [p for p in problems if p.type == ProblemType.MEMORY_PRESSURE]
        assert len(mem) == 1
        assert mem[0].severity == ProblemSeverity.CRITICAL

    def test_memory_pressure_high(self):
        """GPU framebuffer at 92% → MEMORY_PRESSURE HIGH (not CRITICAL)."""
        total = 81920.0
        gpu = make_gpu(0, mem_used_mib=total * 0.92, mem_total_mib=total)
        node = make_node(gpus=[gpu])
        problems = UtilizationAnalyzer().analyze(make_cluster([node]))

        mem = [p for p in problems if p.type == ProblemType.MEMORY_PRESSURE]
        assert len(mem) == 1
        assert mem[0].severity == ProblemSeverity.HIGH

    def test_no_memory_pressure_at_80_pct(self):
        """GPU framebuffer at 80% → no MEMORY_PRESSURE."""
        total = 81920.0
        gpu = make_gpu(0, mem_used_mib=total * 0.80, mem_total_mib=total)
        node = make_node(gpus=[gpu])
        problems = UtilizationAnalyzer().analyze(make_cluster([node]))
        mem = [p for p in problems if p.type == ProblemType.MEMORY_PRESSURE]
        assert len(mem) == 0

    def test_thermal_throttling(self):
        """GPU at 85°C → THERMAL_THROTTLING."""
        gpu = make_gpu(0, temp=85.0)
        node = make_node(gpus=[gpu])
        problems = UtilizationAnalyzer().analyze(make_cluster([node]))
        thermal = [p for p in problems if p.type == ProblemType.THERMAL_THROTTLING]
        assert len(thermal) == 1
        assert thermal[0].metrics["temperature_c"] == 85.0

    def test_no_thermal_below_threshold(self):
        """GPU at 75°C → no thermal warning."""
        gpu = make_gpu(0, temp=75.0)
        node = make_node(gpus=[gpu])
        problems = UtilizationAnalyzer().analyze(make_cluster([node]))
        thermal = [p for p in problems if p.type == ProblemType.THERMAL_THROTTLING]
        assert len(thermal) == 0


# ── Fragmentation Analyzer ───────────────────────────────────────────────────

class TestFragmentationAnalyzer:

    def test_fully_allocated_no_fragmentation(self):
        """8/8 GPUs allocated → no fragmentation."""
        gpus = [make_gpu(i) for i in range(8)]
        node = make_node(gpus=gpus)
        problems = FragmentationAnalyzer().analyze(make_cluster([node]))
        assert len(problems) == 0

    def test_sparse_allocation_flagged(self):
        """2/8 GPUs allocated → GPU_FRAGMENTATION."""
        gpus = []
        for i in range(8):
            if i < 2:
                gpus.append(make_gpu(i, pod_name=f"inf-{i}"))
            else:
                gpus.append(make_gpu(i, pod_name=None))  # unallocated
        node = make_node(gpus=gpus)
        problems = FragmentationAnalyzer().analyze(make_cluster([node]))

        frag = [p for p in problems if p.type == ProblemType.GPU_FRAGMENTATION]
        assert len(frag) == 1
        assert frag[0].severity == ProblemSeverity.MEDIUM  # 2/8 = 0.25, boundary is strict < 0.25
        assert frag[0].metrics["allocated_gpus"] == 2
        assert frag[0].metrics["total_gpus"] == 8

    def test_half_allocated_medium_severity(self):
        """4/8 GPUs allocated → MEDIUM severity (at threshold boundary)."""
        gpus = [make_gpu(i, pod_name=f"pod-{i}" if i < 4 else None) for i in range(8)]
        node = make_node(gpus=gpus)
        # threshold is 0.5; alloc_ratio check is > threshold, so 4/8 = 0.5 IS flagged
        problems = FragmentationAnalyzer().analyze(make_cluster([node]))
        frag = [p for p in problems if p.type == ProblemType.GPU_FRAGMENTATION]
        assert len(frag) == 1
        assert frag[0].severity == ProblemSeverity.MEDIUM

    def test_cost_estimate_in_metrics(self):
        """Fragmentation problem should include a cost estimate."""
        gpus = [make_gpu(i, pod_name=f"pod-{i}" if i < 2 else None) for i in range(8)]
        node = make_node(gpus=gpus)
        problems = FragmentationAnalyzer().analyze(make_cluster([node]))
        frag = [p for p in problems if p.type == ProblemType.GPU_FRAGMENTATION]
        assert "estimated_waste_per_hour_usd" in frag[0].metrics
        assert frag[0].metrics["estimated_waste_per_hour_usd"] > 0


# ── Collective (NCCL Straggler) Analyzer ─────────────────────────────────────

class TestCollectiveAnalyzer:

    def _make_dist_job_cluster(self, straggler_util=15.0, healthy_util=88.0):
        """8-node cluster all in the same distributed job. Node 3 is slow."""
        nodes = []
        for ni in range(8):
            is_straggler = (ni == 3)
            util = straggler_util if is_straggler else healthy_util
            ib   = 1.5 if is_straggler else 170.0

            gpus = [make_gpu(i, util_sm=util, job_name="dist-train", pod_name=f"t-{ni}-{i}")
                    for i in range(8)]
            nodes.append(make_node(f"node-{ni:02d}", gpus, ib_rx=ib, ib_tx=ib))
        return make_cluster(nodes)

    def test_straggler_detected(self):
        """Node running at 17% of job median → NCCL_STRAGGLER CRITICAL."""
        snapshot = self._make_dist_job_cluster(straggler_util=15.0, healthy_util=88.0)
        problems = CollectiveAnalyzer().analyze(snapshot)

        straggler = [p for p in problems if p.type == ProblemType.NCCL_STRAGGLER]
        assert len(straggler) == 1, f"Expected 1 straggler, got {len(straggler)}"
        assert straggler[0].node_name == "node-03"
        assert straggler[0].severity == ProblemSeverity.CRITICAL
        assert straggler[0].job_name == "dist-train"

    def test_ib_degraded_flag_set(self):
        """Low IB bandwidth on straggler should set ib_degraded=True."""
        snapshot = self._make_dist_job_cluster()
        problems = CollectiveAnalyzer().analyze(snapshot)
        straggler = [p for p in problems if p.type == ProblemType.NCCL_STRAGGLER]
        assert straggler[0].metrics["ib_degraded"] is True

    def test_no_straggler_when_all_equal(self):
        """If all nodes run at same utilization, no straggler."""
        nodes = []
        for ni in range(4):
            gpus = [make_gpu(i, util_sm=85.0, job_name="uniform-job", pod_name=f"t-{ni}-{i}")
                    for i in range(8)]
            nodes.append(make_node(f"node-{ni:02d}", gpus, ib_rx=170, ib_tx=170))
        snapshot = make_cluster(nodes)
        problems = CollectiveAnalyzer().analyze(snapshot)
        straggler = [p for p in problems if p.type == ProblemType.NCCL_STRAGGLER]
        assert len(straggler) == 0

    def test_single_node_job_not_flagged(self):
        """A single-node job cannot have a straggler — needs a median to compare against."""
        gpus = [make_gpu(i, util_sm=20.0, job_name="single-node-job", pod_name=f"t-{i}")
                for i in range(8)]
        node = make_node(gpus=gpus, ib_rx=1.0)
        snapshot = make_cluster([node])
        problems = CollectiveAnalyzer().analyze(snapshot)
        straggler = [p for p in problems if p.type == ProblemType.NCCL_STRAGGLER]
        assert len(straggler) == 0

    def test_idle_job_not_flagged(self):
        """If the whole job is idle (starting up), don't flag stragglers."""
        nodes = []
        for ni in range(4):
            gpus = [make_gpu(i, util_sm=2.0 if ni == 0 else 5.0,
                            job_name="cold-job", pod_name=f"t-{ni}-{i}")
                    for i in range(8)]
            nodes.append(make_node(f"node-{ni:02d}", gpus))
        problems = CollectiveAnalyzer().analyze(make_cluster(nodes))
        straggler = [p for p in problems if p.type == ProblemType.NCCL_STRAGGLER]
        assert len(straggler) == 0, "Should not flag stragglers in an idle job"


# ── Topology Analyzer ─────────────────────────────────────────────────────────

class TestTopologyAnalyzer:

    def test_same_nvlink_domain_no_problem(self):
        """GPUs 0,1,2,3 are in the same NVLink domain — no topology mismatch."""
        gpus = [make_gpu(i, job_name="tp-job", pod_name=f"tp-{i}") for i in range(4)]
        # Add 4 unallocated GPUs to make it an 8-GPU node
        gpus += [make_gpu(i, pod_name=None) for i in range(4, 8)]
        node = make_node(gpus=gpus)
        problems = TopologyAnalyzer().analyze(make_cluster([node]))
        topo = [p for p in problems if p.type == ProblemType.TOPOLOGY_MISMATCH]
        assert len(topo) == 0

    def test_cross_domain_flagged(self):
        """GPUs 2,3,4,5 span both NVLink domains → TOPOLOGY_MISMATCH."""
        # Domain A = {0,1,2,3}, Domain B = {4,5,6,7}
        # GPUs 2,3 are in A; GPUs 4,5 are in B → cross-domain
        job_indices = [2, 3, 4, 5]
        gpus = []
        for i in range(8):
            if i in job_indices:
                gpus.append(make_gpu(i, job_name="tp-job", pod_name=f"tp-{i}"))
            else:
                gpus.append(make_gpu(i, pod_name=None))
        node = make_node(gpus=gpus)
        problems = TopologyAnalyzer().analyze(make_cluster([node]))
        topo = [p for p in problems if p.type == ProblemType.TOPOLOGY_MISMATCH]
        assert len(topo) == 1
        assert set(topo[0].gpu_indices) == set(job_indices)

    def test_full_node_job_no_topology_problem(self):
        """A job using all 8 GPUs on a node — no mismatch regardless of layout."""
        gpus = [make_gpu(i, job_name="full-job", pod_name=f"t-{i}") for i in range(8)]
        node = make_node(gpus=gpus)
        problems = TopologyAnalyzer().analyze(make_cluster([node]))
        topo = [p for p in problems if p.type == ProblemType.TOPOLOGY_MISMATCH]
        assert len(topo) == 0


# ── Synthetic Scenario Validation ────────────────────────────────────────────

class TestSyntheticScenarios:
    """
    Validate that each scenario produces the expected problem types
    and that healthy produces zero problems.
    """

    def _problems_of_type(self, scenario: Scenario, ptype: ProblemType):
        snapshot = SyntheticCollector(scenario=scenario, seed=42).collect()
        report   = Analyzer().analyze(snapshot)
        return [p for p in report.problems if p.type == ptype]

    def _all_problems(self, scenario: Scenario):
        snapshot = SyntheticCollector(scenario=scenario, seed=42).collect()
        return Analyzer().analyze(snapshot).problems

    def test_healthy_zero_problems(self):
        """Healthy scenario must produce exactly zero problems."""
        problems = self._all_problems(Scenario.HEALTHY)
        assert len(problems) == 0, (
            f"Healthy scenario should have 0 problems, got {len(problems)}: "
            f"{[p.type.value for p in problems]}"
        )

    def test_fragmentation_scenario(self):
        problems = self._problems_of_type(Scenario.GPU_FRAGMENTATION, ProblemType.GPU_FRAGMENTATION)
        assert len(problems) >= 1, "Fragmentation scenario must detect GPU_FRAGMENTATION"

    def test_cpu_starvation_scenario(self):
        problems = self._problems_of_type(Scenario.CPU_STARVATION, ProblemType.CPU_DATA_STARVATION)
        assert len(problems) >= 1, "CPU starvation scenario must detect CPU_DATA_STARVATION"

    def test_nccl_straggler_scenario(self):
        problems = self._problems_of_type(Scenario.NCCL_STRAGGLER, ProblemType.NCCL_STRAGGLER)
        assert len(problems) == 1, "NCCL straggler scenario must detect exactly 1 straggler"
        assert problems[0].node_name == "gpu-node-03"
        assert problems[0].severity == ProblemSeverity.CRITICAL

    def test_memory_pressure_scenario(self):
        problems = self._problems_of_type(Scenario.MEMORY_PRESSURE, ProblemType.MEMORY_PRESSURE)
        assert len(problems) >= 1, "Memory pressure scenario must detect MEMORY_PRESSURE"

    def test_mixed_has_all_major_types(self):
        """Mixed scenario should contain at least 5 of the 7 problem types."""
        problems = self._all_problems(Scenario.MIXED)
        found_types = {p.type for p in problems}
        expected = {
            ProblemType.NCCL_STRAGGLER,
            ProblemType.CPU_DATA_STARVATION,
            ProblemType.MEMORY_PRESSURE,
            ProblemType.GPU_FRAGMENTATION,
            ProblemType.GPU_IDLE_ALLOCATED,
        }
        missing = expected - found_types
        assert not missing, f"Mixed scenario missing problem types: {missing}"

    def test_mixed_has_critical_problem(self):
        """Mixed scenario must have at least one CRITICAL problem."""
        problems = self._all_problems(Scenario.MIXED)
        criticals = [p for p in problems if p.severity == ProblemSeverity.CRITICAL]
        assert len(criticals) >= 1, "Mixed scenario should have at least one CRITICAL problem"

    def test_report_serializes_to_json(self):
        """AnalysisReport.to_json() must produce valid JSON with expected keys."""
        import json
        snapshot = SyntheticCollector(scenario=Scenario.MIXED, seed=42).collect()
        report   = Analyzer().analyze(snapshot)
        data     = json.loads(report.to_json())

        assert "cluster" in data
        assert "problems" in data
        assert "cluster_stats" in data
        assert isinstance(data["problems"], list)
        assert data["problem_count"] == len(data["problems"])

    def test_analysis_is_fast(self):
        """Analysis of a full 8-node cluster should complete in under 50ms."""
        snapshot = SyntheticCollector(scenario=Scenario.MIXED, seed=42).collect()
        report   = Analyzer().analyze(snapshot)
        assert report.analysis_duration_ms < 50, (
            f"Analysis took {report.analysis_duration_ms:.1f}ms — expected < 50ms"
        )
