"""
gpulens.reporters.cli_reporter
===============================
Rich-formatted terminal output for AnalysisReport.

Sections
--------
  1. Header (cluster name, timestamp)
  2. Cluster overview (GPU count, allocation, avg utilization)
  3. Node heatmap (GPU utilization per node, one cell per GPU)
  4. Problem summary (count by severity)
  5. Problems (sorted by severity, with description / impact / fix)
  6. Footer (analysis duration)
"""
from __future__ import annotations

from typing import List

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from gpulens.models.cluster import ClusterSnapshot, NodeSnapshot
from gpulens.models.problems import AnalysisReport, Problem, ProblemSeverity


class CLIReporter:
    """
    Renders an AnalysisReport to the terminal using Rich.

    Parameters
    ----------
    console : Console | None
        Rich Console to write to. Creates a default console if None.
    verbose : bool
        If True, print raw metric values under each problem.
    no_heatmap : bool
        Skip the node heatmap section (useful for very large clusters).
    """

    def __init__(
        self,
        console: Console | None = None,
        verbose: bool           = False,
        no_heatmap: bool        = False,
    ) -> None:
        self.console    = console or Console()
        self.verbose    = verbose
        self.no_heatmap = no_heatmap

    def print_report(self, report: AnalysisReport) -> None:
        self._header(report)
        self._cluster_overview(report.cluster_snapshot)
        if not self.no_heatmap:
            self._node_heatmap(report.cluster_snapshot)
        self._problem_summary(report)
        self._problems(report.problems)
        self._footer(report)

    # ── Section renderers ───────────────────────────────────────────────────

    def _header(self, report: AnalysisReport) -> None:
        snap  = report.cluster_snapshot
        title = Text()
        title.append("⚡ gpulens", style="bold cyan")
        title.append("  GPU Workload Analyzer", style="dim white")

        self.console.print()
        self.console.print(Panel(
            title,
            subtitle=(
                f"[dim]cluster: [bold]{snap.cluster_name}[/bold]"
                f"  |  {snap.snapshot_time.strftime('%Y-%m-%d %H:%M:%S UTC')}[/dim]"
            ),
            border_style="cyan",
            padding=(0, 2),
        ))

    def _util_bar(self, pct: float, width: int = 16) -> Text:
        filled = int(pct / 100.0 * width)
        bar    = "█" * filled + "░" * (width - filled)
        color  = "green" if pct >= 75 else "yellow" if pct >= 40 else "red"
        t = Text()
        t.append(f"[{bar}] ", style=color)
        t.append(f"{pct:5.1f}%", style="bold " + color)
        return t

    def _cluster_overview(self, snap: ClusterSnapshot) -> None:
        self.console.print()
        self.console.print("[bold cyan]Cluster Overview[/bold cyan]")

        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 3))
        table.add_column(style="dim", width=26)
        table.add_column()

        total     = snap.total_gpus
        allocated = snap.allocated_gpus
        alloc_pct = snap.allocation_ratio * 100
        avg_util  = snap.cluster_avg_utilization

        table.add_row("Nodes",          str(len(snap.nodes)))
        table.add_row("Total GPUs",     f"[bold]{total}[/bold]")
        table.add_row(
            "Allocated GPUs",
            f"[bold]{allocated}[/bold] / {total}  "
            + ("[green]" if alloc_pct > 75 else "[yellow]" if alloc_pct > 40 else "[red]")
            + f"({alloc_pct:.0f}%)"
            + ("[/green]" if alloc_pct > 75 else "[/yellow]" if alloc_pct > 40 else "[/red]"),
        )
        table.add_row("Avg GPU Util",   self._util_bar(avg_util))
        table.add_row("Analysis Window", f"{snap.analysis_window_minutes} min")

        self.console.print(table)

    def _node_heatmap(self, snap: ClusterSnapshot) -> None:
        self.console.print()
        self.console.print("[bold cyan]Node GPU Heatmap[/bold cyan]")
        self.console.print(
            "  [dim]■ ≥75% util   ▪ 40-75%   · <40% (allocated)   _ unallocated[/dim]"
        )
        self.console.print()

        for node in snap.nodes:
            label = Text(f"  {node.node_name:<22}", style="dim")
            cells = Text()

            for gpu in node.gpus:
                if not gpu.is_allocated:
                    cells.append(" _", style="dim")
                elif gpu.utilization_sm_pct >= 75:
                    cells.append(" ■", style="bold green")
                elif gpu.utilization_sm_pct >= 40:
                    cells.append(" ▪", style="bold yellow")
                else:
                    cells.append(" ·", style="bold red")

            # CPU strip
            cpu     = node.cpu_utilization_pct
            cpu_col = "red" if cpu >= 90 else "yellow" if cpu >= 60 else "dim"
            cpu_bar = "█" * min(10, int(cpu / 10))
            cpu_txt = Text(f"   CPU [{cpu_bar:<10}] {cpu:.0f}%", style=cpu_col)

            self.console.print(Text.assemble(label, cells, cpu_txt))

        self.console.print()

    def _problem_summary(self, report: AnalysisReport) -> None:
        total  = len(report.problems)
        by_sev = report.by_severity

        if total == 0:
            self.console.print(Panel(
                "[bold green]✓  No problems detected[/bold green]",
                border_style="green",
            ))
            return

        parts: List[str] = []
        for sev in ProblemSeverity:
            count = len(by_sev[sev.value])
            if count > 0:
                parts.append(f"{sev.icon} [bold]{count}[/bold] {sev.value.lower()}")

        summary = Text()
        summary.append(
            f"⚠  {total} problem{'s' if total != 1 else ''} detected    ",
            style="bold white",
        )
        summary.append("  ".join(parts))

        border = "red" if report.has_critical else "yellow"
        self.console.print(Panel(summary, border_style=border))

    def _problems(self, problems: List[Problem]) -> None:
        if not problems:
            return

        self.console.print()
        self.console.print("[bold cyan]Problems[/bold cyan]")
        self.console.print()

        for i, p in enumerate(problems, start=1):
            sev_style = p.severity.rich_style
            icon      = p.severity.icon

            # ── Header ──────────────────────────────────────────────────────
            hdr = Text()
            hdr.append(f"{icon} [{i}] ", style=sev_style)
            hdr.append(p.type.value, style="bold white")
            hdr.append(f"  {p.node_name}", style="dim")
            if p.pod_name:
                hdr.append(f"  pod/{p.pod_name}", style="dim cyan")
            if p.job_name:
                hdr.append(f"  job/{p.job_name}", style="dim magenta")
            if p.gpu_indices:
                indices_str = (
                    str(p.gpu_indices)
                    if len(p.gpu_indices) <= 6
                    else f"[{p.gpu_indices[0]}…{p.gpu_indices[-1]}]"
                )
                hdr.append(f"  GPUs:{indices_str}", style="dim")
            self.console.print(hdr)

            # ── Description ─────────────────────────────────────────────────
            self.console.print(f"   [white]{p.description}[/white]")

            # ── Impact ──────────────────────────────────────────────────────
            self.console.print(f"   [dim]Impact[/dim]  [yellow]{p.impact}[/yellow]")

            # ── Fix ─────────────────────────────────────────────────────────
            for line in p.recommendation.splitlines():
                prefix = "   [dim]Fix[/dim]    " if line == p.recommendation.splitlines()[0] else "           "
                self.console.print(f"{prefix}[green]{line}[/green]")

            # ── Metrics (verbose) ────────────────────────────────────────────
            if self.verbose and p.metrics:
                metrics_str = "  ".join(f"[dim]{k}=[/dim]{v}" for k, v in p.metrics.items())
                self.console.print(f"   [dim]Metrics  {metrics_str}[/dim]")

            self.console.print()

    def _footer(self, report: AnalysisReport) -> None:
        self.console.print(
            f"[dim]Analysis completed in {report.analysis_duration_ms:.1f} ms  "
            f"|  gpulens v0.1.0[/dim]"
        )
        self.console.print()
