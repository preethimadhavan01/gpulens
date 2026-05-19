"""
gpulens.cli
===========
Command-line interface for gpulens.

Commands
--------
  gpulens simulate   — Analyze a synthetic cluster scenario (no real cluster needed)
  gpulens scan       — Analyze a live cluster via Prometheus + DCGM
  gpulens scenarios  — List all available synthetic scenarios
"""
from __future__ import annotations

import json
import sys

import click
from rich.console import Console
from rich.table import Table
from rich import box


# ── CLI group ──────────────────────────────────────────────────────────────

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option("0.1.0", prog_name="gpulens")
def cli() -> None:
    """
    gpulens — GPU workload analyzer for AI infrastructure teams.

    Detects performance and efficiency problems in GPU clusters running on
    Kubernetes: idle allocated GPUs, CPU data-pipeline starvation, NCCL
    collective stragglers, NVLink topology mismatches, and more.
    """
    pass


# ── simulate ───────────────────────────────────────────────────────────────

@cli.command()
@click.option(
    "--scenario", "-s",
    type=click.Choice(
        ["healthy", "fragmentation", "cpu_starvation",
         "nccl_straggler", "cold_start", "memory_pressure", "mixed"],
        case_sensitive=False,
    ),
    default="mixed",
    show_default=True,
    help="Synthetic scenario to generate.",
)
@click.option("--nodes", "-n", default=8, show_default=True,
              help="Number of simulated GPU nodes.")
@click.option("--gpus-per-node", "-g", default=8, show_default=True,
              help="GPUs per node.")
@click.option(
    "--output", "-o",
    type=click.Choice(["cli", "json"]),
    default="cli", show_default=True,
    help="Output format.",
)
@click.option("--verbose", "-v", is_flag=True,
              help="Include raw metric values in problem output.")
@click.option("--seed", default=42, show_default=True,
              help="Random seed for reproducibility.")
@click.option("--no-heatmap", is_flag=True,
              help="Skip the node utilization heatmap.")
def simulate(
    scenario: str,
    nodes: int,
    gpus_per_node: int,
    output: str,
    verbose: bool,
    seed: int,
    no_heatmap: bool,
) -> None:
    """
    Run problem analysis against a synthetic cluster snapshot.

    No live cluster required — useful for demos, CI, and offline development.
    Each scenario embeds known problems so you can validate analyzer behaviour.

    \b
    Examples
    --------
      gpulens simulate
      gpulens simulate --scenario nccl_straggler --nodes 8
      gpulens simulate --scenario fragmentation --verbose
      gpulens simulate --scenario mixed -o json | jq '.problems[0]'
    """
    from gpulens.collectors.synthetic import SyntheticCollector, Scenario
    from gpulens.analyzers import Analyzer
    from gpulens.reporters.cli_reporter import CLIReporter

    collector = SyntheticCollector(
        scenario       = Scenario(scenario),
        node_count     = nodes,
        gpus_per_node  = gpus_per_node,
        seed           = seed,
    )
    snapshot = collector.collect()
    report   = Analyzer().analyze(snapshot)

    if output == "json":
        click.echo(report.to_json())
    else:
        CLIReporter(
            console    = Console(),
            verbose    = verbose,
            no_heatmap = no_heatmap,
        ).print_report(report)


# ── scan ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--prometheus-url", "-p", required=True,
              help="Prometheus endpoint (e.g. http://prometheus.monitoring.svc:9090).")
@click.option("--cluster-name", "-c", default="production", show_default=True,
              help="Cluster label for the report.")
@click.option(
    "--output", "-o",
    type=click.Choice(["cli", "json"]),
    default="cli", show_default=True,
)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--no-heatmap", is_flag=True)
@click.option("--token", default=None,
              help="Bearer token for authenticated Prometheus endpoints.")
@click.option("--no-tls-verify", is_flag=True,
              help="Disable TLS certificate verification.")
def scan(
    prometheus_url: str,
    cluster_name: str,
    output: str,
    verbose: bool,
    no_heatmap: bool,
    token: str | None,
    no_tls_verify: bool,
) -> None:
    """
    Scan a live cluster via Prometheus + DCGM Exporter.

    \b
    Prerequisites
    -------------
      DCGM Exporter:
        helm install dcgm gpu-helm-charts/dcgm-exporter -n monitoring
      kube-state-metrics:
        helm install ksm prometheus-community/kube-state-metrics -n monitoring

    \b
    Examples
    --------
      gpulens scan -p http://prometheus.monitoring.svc:9090
      gpulens scan -p http://localhost:9090 -c prod -o json
      gpulens scan -p https://prom.internal --token $TOKEN
    """
    from gpulens.collectors.prometheus import PrometheusCollector
    from gpulens.analyzers import Analyzer
    from gpulens.reporters.cli_reporter import CLIReporter

    console = Console()

    collector = PrometheusCollector(
        prometheus_url = prometheus_url,
        cluster_name   = cluster_name,
        tls_verify     = not no_tls_verify,
        bearer_token   = token,
    )

    if not collector.is_available():
        console.print(f"[red]✗  Cannot reach Prometheus at {prometheus_url}[/red]")
        console.print("[dim]   Check the URL and ensure Prometheus is reachable.[/dim]")
        sys.exit(1)

    console.print(f"[dim]Collecting metrics from {prometheus_url} …[/dim]")

    try:
        snapshot = collector.collect()
    except Exception as exc:
        console.print(f"[red]✗  Metric collection failed: {exc}[/red]")
        sys.exit(1)

    if not snapshot.nodes:
        console.print("[yellow]⚠  No GPU nodes found in Prometheus data.[/yellow]")
        console.print("[dim]   Ensure DCGM Exporter is running and Prometheus is scraping it.[/dim]")
        sys.exit(1)

    report = Analyzer().analyze(snapshot)

    if output == "json":
        click.echo(report.to_json())
    else:
        CLIReporter(
            console    = console,
            verbose    = verbose,
            no_heatmap = no_heatmap,
        ).print_report(report)


# ── scenarios ──────────────────────────────────────────────────────────────

@cli.command()
def scenarios() -> None:
    """List all available synthetic scenarios with expected problems."""
    console = Console()

    table = Table(
        title       = "gpulens Synthetic Scenarios",
        box         = box.ROUNDED,
        border_style= "cyan",
        show_lines  = True,
    )
    table.add_column("Scenario",         style="bold cyan", width=18)
    table.add_column("Problems Embedded", width=50)
    table.add_column("Best Used For",    style="dim", width=30)

    rows = [
        (
            "healthy",
            "[green]None — all GPUs at >88% utilization[/green]",
            "False-positive validation",
        ),
        (
            "fragmentation",
            "[yellow]GPU_FRAGMENTATION on 50% of nodes[/yellow]\n"
            "(2/8 GPUs allocated, rest idle)",
            "Bin-packing & consolidation",
        ),
        (
            "cpu_starvation",
            "[orange1]CPU_DATA_STARVATION cluster-wide[/orange1]\n"
            "(GPU 35% util, CPU 97%)",
            "DataLoader tuning",
        ),
        (
            "nccl_straggler",
            "[red]NCCL_STRAGGLER on node-03[/red]\n"
            "(IB degraded to ~1 Gbps vs 175 Gbps)",
            "Network fault detection",
        ),
        (
            "cold_start",
            "[yellow]GPU_IDLE_ALLOCATED on nodes 0-3[/yellow]\n"
            "(pods Running but no GPU activity)",
            "Startup latency analysis",
        ),
        (
            "memory_pressure",
            "[orange1]MEMORY_PRESSURE on all nodes[/orange1]\n"
            "(92-99% GPU framebuffer used)",
            "OOM prevention / batch sizing",
        ),
        (
            "mixed",
            "[red]All problem types across 8 nodes[/red]\n"
            "(realistic production cluster)",
            "Full demo / integration testing",
        ),
    ]

    for name, problems, use_case in rows:
        table.add_row(name, problems, use_case)

    console.print()
    console.print(table)
    console.print()
    console.print("[dim]Run a scenario: gpulens simulate --scenario <name>[/dim]")
    console.print()
