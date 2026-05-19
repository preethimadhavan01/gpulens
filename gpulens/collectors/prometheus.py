"""
gpulens.collectors.prometheus
==============================
Collector for live clusters via Prometheus + DCGM Exporter + kube-state-metrics.

Prerequisites
-------------
  * NVIDIA DCGM Exporter running as a DaemonSet on GPU nodes
  * Prometheus scraping DCGM on port 9400 (default)
  * kube-state-metrics for pod → node GPU attribution
  * Optional: node_exporter for host CPU/memory metrics

Expected Prometheus metrics
---------------------------
  DCGM_FI_DEV_GPU_UTIL          SM utilization 0-100
  DCGM_FI_DEV_MEM_COPY_UTIL    Memory bandwidth utilization 0-100
  DCGM_FI_DEV_FB_USED           Framebuffer used (MiB)
  DCGM_FI_DEV_FB_FREE           Framebuffer free (MiB)
  DCGM_FI_DEV_SM_CLOCK          SM clock (MHz)
  DCGM_FI_DEV_MEM_CLOCK         Memory clock (MHz)
  DCGM_FI_DEV_POWER_USAGE       Power draw (W)
  DCGM_FI_DEV_GPU_TEMP          Temperature (°C)
  kube_pod_container_resource_requests{resource="nvidia.com/gpu"}
  node_cpu_seconds_total (for host CPU utilization)

Helm quickstart (DCGM + kube-state-metrics)
--------------------------------------------
  helm repo add gpu-helm-charts https://nvidia.github.io/dcgm-exporter/helm-charts
  helm install dcgm-exporter gpu-helm-charts/dcgm-exporter -n monitoring

  helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
  helm install kube-state-metrics prometheus-community/kube-state-metrics -n monitoring
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from gpulens.collectors.base import BaseCollector
from gpulens.models.cluster import (
    ClusterSnapshot,
    GPUMetrics,
    GPUType,
    NodeSnapshot,
)

# How DCGM labels the GPU index field (may differ by exporter version)
_GPU_INDEX_LABELS = ("gpu", "GPU_I_ID", "gpuIndex")


def _get_gpu_index(metric_labels: Dict[str, str]) -> int:
    for label in _GPU_INDEX_LABELS:
        if label in metric_labels:
            try:
                return int(metric_labels[label])
            except ValueError:
                pass
    return 0


def _get_node_name(metric_labels: Dict[str, str]) -> str:
    """DCGM may use 'Hostname', 'instance', or 'node' for the host label."""
    for label in ("Hostname", "node", "instance"):
        if label in metric_labels:
            return metric_labels[label].split(":")[0]  # strip port from instance
    return "unknown"


class PrometheusCollector(BaseCollector):
    """
    Collects a ClusterSnapshot from a live Prometheus endpoint.

    Usage
    -----
    >>> collector = PrometheusCollector("http://prometheus.monitoring.svc:9090")
    >>> if collector.is_available():
    ...     snapshot = collector.collect()
    """

    def __init__(
        self,
        prometheus_url: str,
        cluster_name: str = "production",
        cluster_label: Optional[str] = None,
        namespace: Optional[str] = None,
        timeout_seconds: int = 15,
        tls_verify: bool = True,
        bearer_token: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        cluster_label : Optional[str]
            Prometheus label that distinguishes clusters in a federated /
            Thanos / Cortex setup (typically "cluster"). When set, every
            query is filtered to {<cluster_label>="<cluster_name>"} so that
            metrics from other clusters don't bleed into this collection.
            Leave as None for single-cluster Prometheus deployments.
        namespace : Optional[str]
            Restrict pod attribution to a single Kubernetes namespace
            (multi-tenant clusters). DCGM-level GPU metrics still come
            through, but only pods in this namespace are mapped onto GPUs.
        """
        self.prometheus_url = prometheus_url.rstrip("/")
        self.cluster_name   = cluster_name
        self.cluster_label  = cluster_label
        self.namespace      = namespace
        self.timeout        = timeout_seconds

        self._session = requests.Session()
        self._session.verify = tls_verify
        if bearer_token:
            self._session.headers["Authorization"] = f"Bearer {bearer_token}"

    def is_available(self) -> bool:
        try:
            r = self._session.get(f"{self.prometheus_url}/-/healthy", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    # ── Prometheus query helpers ────────────────────────────────────────────

    def _metric(
        self,
        name: str,
        extra: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Render a metric selector `name{...}` with cluster + tenant filters
        applied. `extra` is additional exact-match selectors to AND in.
        """
        parts: List[str] = []
        if self.cluster_label:
            parts.append(f'{self.cluster_label}="{self.cluster_name}"')
        if extra:
            parts.extend(f'{k}="{v}"' for k, v in extra.items())
        return f"{name}{{{','.join(parts)}}}" if parts else name

    def _query(self, promql: str) -> List[Dict[str, Any]]:
        resp = self._session.get(
            f"{self.prometheus_url}/api/v1/query",
            params={"query": promql},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "success":
            return []
        return data.get("data", {}).get("result", [])

    def _scalar_map(
        self,
        promql: str,
        key_labels: tuple[str, ...] = ("Hostname", "gpu"),
    ) -> Dict[tuple, float]:
        """
        Returns {(node_name, gpu_index): value} from an instant query.
        """
        result = {}
        for item in self._query(promql):
            m    = item["metric"]
            node = _get_node_name(m)
            idx  = _get_gpu_index(m)
            try:
                result[(node, idx)] = float(item["value"][1])
            except (IndexError, ValueError):
                pass
        return result

    # ── Collection ──────────────────────────────────────────────────────────

    def collect(self) -> ClusterSnapshot:
        # ── DCGM metrics ───────────────────────────────────────────────────
        sm_util  = self._scalar_map(self._metric("DCGM_FI_DEV_GPU_UTIL"))
        mem_util = self._scalar_map(self._metric("DCGM_FI_DEV_MEM_COPY_UTIL"))
        fb_used  = self._scalar_map(self._metric("DCGM_FI_DEV_FB_USED"))
        fb_free  = self._scalar_map(self._metric("DCGM_FI_DEV_FB_FREE"))
        sm_clock = self._scalar_map(self._metric("DCGM_FI_DEV_SM_CLOCK"))
        mem_clk  = self._scalar_map(self._metric("DCGM_FI_DEV_MEM_CLOCK"))
        power    = self._scalar_map(self._metric("DCGM_FI_DEV_POWER_USAGE"))
        temp     = self._scalar_map(self._metric("DCGM_FI_DEV_GPU_TEMP"))

        # ── Pod attribution (kube-state-metrics) ───────────────────────────
        # kube_pod_container_resource_requests has node, namespace, pod labels
        pod_filters: Dict[str, str] = {"resource": "nvidia.com/gpu"}
        if self.namespace:
            pod_filters["namespace"] = self.namespace
        pod_allocs = self._query(
            self._metric("kube_pod_container_resource_requests", pod_filters) + " > 0"
        )
        # node → list of pod dicts
        node_pods: Dict[str, List[Dict]] = {}
        for item in pod_allocs:
            m = item["metric"]
            node_pods.setdefault(m.get("node", "unknown"), []).append({
                "pod":       m.get("pod", ""),
                "namespace": m.get("namespace", ""),
                "job":       m.get("label_batch_kubernetes_io_job_name", ""),
                "gpu_count": int(float(item["value"][1])),
            })

        # ── Host CPU (node_exporter) ────────────────────────────────────────
        # node_exporter labels series with instance=<pod_ip>:9100, while DCGM
        # and kube-state-metrics use the K8s node name. Bridge the gap via
        # kube_node_info, which carries both `node` and `internal_ip`.
        ip_to_node: Dict[str, str] = {}
        for item in self._query(self._metric("kube_node_info")):
            m = item["metric"]
            node = m.get("node")
            ip   = m.get("internal_ip")
            if node and ip:
                ip_to_node[ip] = node

        # rate over 2m window — fallback to 0 if not available
        cpu_idle_map: Dict[str, float] = {}
        cpu_metric = self._metric("node_cpu_seconds_total", {"mode": "idle"})
        for item in self._query(
            f"avg by (instance) (rate({cpu_metric}[2m]))"
        ):
            instance = item["metric"].get("instance", "")
            host = instance.split(":")[0]
            node = ip_to_node.get(host, host)
            try:
                cpu_idle_map[node] = float(item["value"][1])
            except (IndexError, ValueError):
                pass

        # ── Aggregate into NodeSnapshots ────────────────────────────────────
        all_keys = set(sm_util.keys()) | set(fb_used.keys())
        node_to_indices: Dict[str, set] = {}
        for (node, idx) in all_keys:
            node_to_indices.setdefault(node, set()).add(idx)

        nodes: List[NodeSnapshot] = []
        for node_name, indices in sorted(node_to_indices.items()):
            pods_on_node = node_pods.get(node_name, [])
            # Simple attribution: assign pods round-robin to GPU indices
            pod_iter = iter(pods_on_node)
            pod_by_gpu: Dict[int, Dict] = {}
            for idx in sorted(indices):
                try:
                    pod_by_gpu[idx] = next(pod_iter)
                except StopIteration:
                    pass

            gpus: List[GPUMetrics] = []
            for idx in sorted(indices):
                key  = (node_name, idx)
                pod  = pod_by_gpu.get(idx)
                used = fb_used.get(key, 0.0)
                free = fb_free.get(key, 0.0)
                total = used + free if (used + free) > 0 else 40960.0

                gpus.append(GPUMetrics(
                    gpu_index              = idx,
                    uuid                   = f"GPU-{uuid.uuid4().hex[:8].upper()}",
                    utilization_sm_pct     = sm_util.get(key, 0.0),
                    utilization_memory_pct = mem_util.get(key, 0.0),
                    memory_used_mib        = used,
                    memory_total_mib       = total,
                    sm_clock_mhz           = sm_clock.get(key, 0.0),
                    memory_clock_mhz       = mem_clk.get(key, 0.0),
                    power_watts            = power.get(key, 0.0),
                    temperature_c          = temp.get(key, 0.0),
                    pod_name               = pod["pod"] if pod else None,
                    namespace              = pod["namespace"] if pod else None,
                    job_name               = pod.get("job") if pod else None,
                ))

            cpu_idle = cpu_idle_map.get(node_name, 0.0)
            cpu_util = max(0.0, (1.0 - cpu_idle) * 100.0)

            nodes.append(NodeSnapshot(
                node_name              = node_name,
                gpu_count              = len(indices),
                gpu_type               = GPUType.UNKNOWN,
                gpus                   = gpus,
                cpu_utilization_pct    = round(cpu_util, 1),
                memory_utilization_pct = 0.0,  # requires node_exporter MemAvailable metric
                network_rx_gbps        = 0.0,
                network_tx_gbps        = 0.0,
                infiniband_rx_gbps     = 0.0,  # requires infiniband_exporter
                infiniband_tx_gbps     = 0.0,
            ))

        return ClusterSnapshot(
            cluster_name   = self.cluster_name,
            nodes          = nodes,
            snapshot_time  = datetime.utcnow(),
        )
