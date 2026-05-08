"""Rich-powered progress visualization for the diffusion text pipeline."""

import time
from collections import deque

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
)
from rich.layout import Layout
from rich.text import Text
from rich.columns import Columns
from rich import box

console = Console()

BANNER = r"""[bold cyan]
    ╔══════════════════════════════════════════════════════════╗
    ║   ░█▀▄░▀█▀░█▀▀░█▀▀░█░█░█▀▀░▀█▀░█▀█░█▀█               ║
    ║   ░█░█░░█░░█▀▀░█▀▀░█░█░▀▀█░░█░░█░█░█░█               ║
    ║   ░▀▀░░▀▀▀░▀░░░▀░░░▀▀▀░▀▀▀░▀▀▀░▀▀▀░▀░▀               ║
    ║          [bold white]D3PM Discrete Diffusion for Text[/bold white]             ║
    ╚══════════════════════════════════════════════════════════╝
[/bold cyan]"""


def print_banner():
    console.print(BANNER)


def make_header_panel(title, subtitle="", style="bold cyan"):
    content = Text(title, style=style, justify="center")
    if subtitle:
        content.append(f"\n{subtitle}", style="dim")
    return Panel(content, box=box.DOUBLE_EDGE, border_style="cyan", padding=(0, 2))


def make_download_progress():
    return Progress(
        SpinnerColumn("dots12", style="cyan"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(
            bar_width=40,
            style="cyan",
            complete_style="bold green",
            finished_style="bold green",
        ),
        MofNCompleteColumn(),
        TextColumn("[dim]•[/dim]"),
        TextColumn("[yellow]{task.fields[chars]}[/yellow]"),
        TextColumn("[dim]•[/dim]"),
        TimeElapsedColumn(),
        TextColumn("[dim]•[/dim]"),
        TextColumn("[magenta]{task.fields[rate]}[/magenta]"),
        console=console,
        transient=False,
    )


def make_tokenize_progress():
    return Progress(
        SpinnerColumn("dots12", style="cyan"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(
            bar_width=40,
            style="cyan",
            complete_style="bold green",
            finished_style="bold green",
        ),
        MofNCompleteColumn(),
        TextColumn("[dim]•[/dim]"),
        TimeElapsedColumn(),
        TextColumn("[dim]•[/dim]"),
        TextColumn("[yellow]{task.fields[seqs]} seqs[/yellow]"),
        console=console,
        transient=False,
    )


def make_eval_progress():
    return Progress(
        SpinnerColumn("point", style="cyan"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(
            bar_width=40,
            style="cyan",
            complete_style="bold green",
            finished_style="bold green",
        ),
        MofNCompleteColumn(),
        TextColumn("[dim]•[/dim]"),
        TimeElapsedColumn(),
        TextColumn("[dim]•[/dim]"),
        TextColumn("[yellow]{task.fields[loss]}[/yellow]"),
        console=console,
        transient=False,
    )


# ── Training dashboard ──────────────────────────────────────

SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(values, width=20):
    if not values:
        return ""
    recent = list(values)[-width:]
    mn, mx = min(recent), max(recent)
    rng = mx - mn if mx != mn else 1.0
    return "".join(SPARKLINE_CHARS[min(int((v - mn) / rng * 7), 7)] for v in recent)


class TrainingDashboard:
    """Rich Live display for training progress with sparklines and stats."""

    def __init__(
        self, max_steps, device_name, amp_enabled, model_params, run_dir, start_step=0
    ):
        self.max_steps = max_steps
        self.device_name = device_name
        self.amp_enabled = amp_enabled
        self.model_params = model_params
        self.run_dir = run_dir
        self.step = start_step
        self.start_time = time.time()
        self.loss_history = deque(maxlen=100)
        self.lr_history = deque(maxlen=100)
        self.grad_norm_history = deque(maxlen=100)
        self.current_loss = 0.0
        self.current_lr = 0.0
        self.current_grad_norm = 0.0
        self.best_val_loss = float("inf")
        self.last_val_loss = None
        self.last_val_acc = None
        self.steps_per_sec = 0.0
        self._last_step_time = time.time()
        self._step_times = deque(maxlen=50)
        self.live = Live(self._render(), console=console, refresh_per_second=4)

    def __enter__(self):
        self.live.__enter__()
        return self

    def __exit__(self, *args):
        self.live.__exit__(*args)

    def update(self, step, loss=None, lr=None, grad_norm=None):
        self.step = step
        now = time.time()
        dt = now - self._last_step_time
        self._last_step_time = now
        if dt > 0:
            self._step_times.append(dt)
            self.steps_per_sec = 1.0 / (sum(self._step_times) / len(self._step_times))

        if loss is not None:
            self.current_loss = loss
            self.loss_history.append(loss)
        if lr is not None:
            self.current_lr = lr
            self.lr_history.append(lr)
        if grad_norm is not None:
            self.current_grad_norm = grad_norm
            self.grad_norm_history.append(grad_norm)

        self.live.update(self._render())

    def update_val(self, val_loss, val_acc):
        self.last_val_loss = val_loss
        self.last_val_acc = val_acc
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
        self.live.update(self._render())

    def _render(self):
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        # Header
        pct = self.step / max(1, self.max_steps) * 100
        header_text = Text.assemble(
            ("D3PM Training", "bold cyan"),
            ("  │  ", "dim"),
            (f"Step {self.step:,}/{self.max_steps:,}", "bold white"),
            ("  │  ", "dim"),
            (f"{pct:.1f}%", "bold green" if pct > 0 else "dim"),
        )
        layout["header"].update(Panel(header_text, box=box.HEAVY, border_style="cyan"))

        # Body: two columns
        body_layout = Layout()
        body_layout.split_row(
            Layout(name="metrics", ratio=3),
            Layout(name="charts", ratio=2),
        )

        # Metrics table
        metrics = Table(
            box=box.SIMPLE_HEAVY, show_header=False, padding=(0, 1), border_style="blue"
        )
        metrics.add_column("Key", style="bold", width=16)
        metrics.add_column("Value", width=28)

        # Loss with trend arrow
        if len(self.loss_history) >= 2:
            trend = self.loss_history[-1] - self.loss_history[-2]
            arrow = (
                "[green]↓[/green]"
                if trend < 0
                else "[red]↑[/red]"
                if trend > 0
                else "[dim]→[/dim]"
            )
        else:
            arrow = ""
        metrics.add_row(
            "Train Loss", f"[bold yellow]{self.current_loss:.4f}[/bold yellow] {arrow}"
        )
        metrics.add_row("Learning Rate", f"[cyan]{self.current_lr:.2e}[/cyan]")
        # Grad norm with color coding: green < 1, yellow 1-10, red > 10
        gn = self.current_grad_norm
        if gn > 0:
            gn_style = "green" if gn < 1.0 else "yellow" if gn < 10.0 else "bold red"
            gn_warning = " [bold red]⚠[/bold red]" if gn > 50.0 else ""
            metrics.add_row("Grad Norm", f"[{gn_style}]{gn:.2f}[/{gn_style}]{gn_warning}")
        metrics.add_row("Speed", f"[green]{self.steps_per_sec:.1f}[/green] steps/s")

        elapsed = time.time() - self.start_time
        eta = (self.max_steps - self.step) / max(0.01, self.steps_per_sec)
        metrics.add_row("Elapsed", f"[white]{_fmt_time(elapsed)}[/white]")
        metrics.add_row("ETA", f"[white]{_fmt_time(eta)}[/white]")
        metrics.add_row("Device", f"[magenta]{self.device_name}[/magenta]")
        metrics.add_row(
            "AMP",
            "[green]✓ enabled[/green]"
            if self.amp_enabled
            else "[yellow]✗ disabled[/yellow]",
        )
        metrics.add_row("Parameters", f"[white]{self.model_params:,}[/white]")

        if self.last_val_loss is not None:
            best_marker = (
                " [green]★[/green]" if self.last_val_loss <= self.best_val_loss else ""
            )
            metrics.add_row(
                "Val Loss", f"[bold]{self.last_val_loss:.4f}[/bold]{best_marker}"
            )
        if self.last_val_acc is not None:
            metrics.add_row("Val Recon Acc", f"[bold]{self.last_val_acc:.2%}[/bold]")
        if self.best_val_loss < float("inf"):
            metrics.add_row(
                "Best Val Loss", f"[green bold]{self.best_val_loss:.4f}[/green bold]"
            )

        layout["body"].update(body_layout)
        body_layout["metrics"].update(
            Panel(
                metrics,
                title="[bold]Metrics[/bold]",
                border_style="blue",
                box=box.ROUNDED,
            )
        )

        # Sparkline charts
        chart_lines = []
        if self.loss_history:
            loss_spark = sparkline(self.loss_history, width=30)
            mn = min(self.loss_history)
            mx = max(self.loss_history)
            chart_lines.append(f"[bold]Loss[/bold]  [yellow]{loss_spark}[/yellow]")
            chart_lines.append(f"       [dim]{mn:.2f} ── {mx:.2f}[/dim]\n")
        if self.lr_history:
            lr_spark = sparkline(self.lr_history, width=30)
            chart_lines.append(f"[bold]LR[/bold]    [cyan]{lr_spark}[/cyan]")
            mn = min(self.lr_history)
            mx = max(self.lr_history)
            chart_lines.append(f"       [dim]{mn:.2e} ── {mx:.2e}[/dim]\n")
        if self.grad_norm_history:
            gn_spark = sparkline(self.grad_norm_history, width=30)
            gn_color = "red" if max(self.grad_norm_history) > 10 else "green"
            chart_lines.append(f"[bold]∇Norm[/bold] [{gn_color}]{gn_spark}[/{gn_color}]")
            mn = min(self.grad_norm_history)
            mx = max(self.grad_norm_history)
            chart_lines.append(f"       [dim]{mn:.2f} ── {mx:.2f}[/dim]")

        charts_text = (
            "\n".join(chart_lines) if chart_lines else "[dim]Waiting for data...[/dim]"
        )
        body_layout["charts"].update(
            Panel(
                charts_text,
                title="[bold]Trends[/bold]",
                border_style="blue",
                box=box.ROUNDED,
            )
        )

        # Footer: progress bar
        pct_val = self.step / max(1, self.max_steps)
        bar_width = 50
        filled = int(pct_val * bar_width)
        bar = f"[bold green]{'━' * filled}[/bold green][dim]{'─' * (bar_width - filled)}[/dim]"
        footer_text = Text.from_markup(f"  {bar}  [bold]{pct:.1f}%[/bold]")
        layout["footer"].update(Panel(footer_text, box=box.HEAVY, border_style="green"))

        return layout


def _fmt_time(seconds):
    if seconds < 0:
        return "???"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"


# ── Pipeline tracker ────────────────────────────────────────

STAGE_ICONS = {
    "pending": "[dim]○[/dim]",
    "running": "[bold yellow]◉[/bold yellow]",
    "done": "[bold green]✓[/bold green]",
    "skipped": "[dim blue]⊘[/dim blue]",
    "failed": "[bold red]✗[/bold red]",
}


class PipelineTracker:
    """Tracks and displays pipeline stage progress."""

    def __init__(self, stages):
        self.stages = []
        for s in stages:
            self.stages.append(
                {
                    "name": s,
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                }
            )
        self.pipeline_start = time.time()

    def start_stage(self, idx):
        self.stages[idx]["status"] = "running"
        self.stages[idx]["start_time"] = time.time()
        self._print()

    def complete_stage(self, idx):
        self.stages[idx]["status"] = "done"
        self.stages[idx]["end_time"] = time.time()
        self._print()

    def skip_stage(self, idx):
        self.stages[idx]["status"] = "skipped"
        self._print()

    def fail_stage(self, idx):
        self.stages[idx]["status"] = "failed"
        self.stages[idx]["end_time"] = time.time()
        self._print()

    def _print(self):
        table = Table(
            box=box.ROUNDED,
            border_style="cyan",
            show_header=False,
            padding=(0, 1),
            title="[bold cyan]Pipeline Progress[/bold cyan]",
        )
        table.add_column("", width=3)
        table.add_column("Stage", width=36)
        table.add_column("Time", width=12, justify="right")

        for i, s in enumerate(self.stages):
            icon = STAGE_ICONS[s["status"]]
            name = s["name"]
            if s["status"] == "running":
                name = f"[bold yellow]{name}[/bold yellow]"
                elapsed = time.time() - s["start_time"]
                time_str = f"[yellow]{_fmt_time(elapsed)}[/yellow]"
            elif s["status"] == "done":
                name = f"[green]{name}[/green]"
                elapsed = s["end_time"] - s["start_time"]
                time_str = f"[green]{_fmt_time(elapsed)}[/green]"
            elif s["status"] == "skipped":
                name = f"[dim]{name}[/dim]"
                time_str = "[dim]skipped[/dim]"
            elif s["status"] == "failed":
                name = f"[red]{name}[/red]"
                elapsed = s["end_time"] - s["start_time"]
                time_str = f"[red]{_fmt_time(elapsed)}[/red]"
            else:
                name = f"[dim]{name}[/dim]"
                time_str = ""

            table.add_row(icon, name, time_str)

        total_elapsed = time.time() - self.pipeline_start
        done_count = sum(1 for s in self.stages if s["status"] in ("done", "skipped"))
        total = len(self.stages)

        console.clear()
        print_banner()
        console.print(table)
        console.print(
            f"\n  [dim]Progress:[/dim] [bold]{done_count}/{total}[/bold] stages  "
            f"[dim]│[/dim]  [dim]Total:[/dim] [bold]{_fmt_time(total_elapsed)}[/bold]\n"
        )

    def finish(self):
        total = time.time() - self.pipeline_start
        done = sum(1 for s in self.stages if s["status"] == "done")
        skipped = sum(1 for s in self.stages if s["status"] == "skipped")
        failed = sum(1 for s in self.stages if s["status"] == "failed")

        self._print()

        summary = Table(
            box=box.DOUBLE_EDGE,
            border_style="green" if failed == 0 else "red",
            title="[bold]Pipeline Complete[/bold]",
        )
        summary.add_column("Metric", style="bold")
        summary.add_column("Value")
        summary.add_row("Total Time", _fmt_time(total))
        summary.add_row("Completed", f"[green]{done}[/green]")
        summary.add_row("Skipped", f"[blue]{skipped}[/blue]")
        if failed > 0:
            summary.add_row("Failed", f"[bold red]{failed}[/bold red]")
        console.print(summary)


# ── Sampling display ────────────────────────────────────────


def print_sample(idx, text, total, prefix=None):
    title = f"Sample {idx}/{total}"
    if prefix:
        title += f'  [dim]prefix: "{prefix}"[/dim]'

    panel = Panel(
        Text(text, style="white"),
        title=f"[bold cyan]{title}[/bold cyan]",
        border_style="blue",
        box=box.ROUNDED,
        padding=(1, 2),
    )
    console.print(panel)


def print_eval_results(results):
    table = Table(
        title="[bold cyan]Evaluation Results[/bold cyan]",
        box=box.DOUBLE_EDGE,
        border_style="cyan",
        show_header=True,
        padding=(0, 2),
    )
    table.add_column("Metric", style="bold white")
    table.add_column("Value", justify="right")

    for k, v in results.items():
        if isinstance(v, float):
            if "accuracy" in k.lower():
                val_str = f"[bold green]{v:.2%}[/bold green]"
            elif "loss" in k.lower():
                val_str = f"[bold yellow]{v:.6f}[/bold yellow]"
            else:
                val_str = f"{v:.6f}"
        else:
            val_str = f"[white]{v:,}[/white]" if isinstance(v, int) else str(v)
        table.add_row(k.replace("_", " ").title(), val_str)

    console.print()
    console.print(table)
    console.print()


def print_stage_header(stage_num, total, description):
    text = Text.assemble(
        (f"  Stage {stage_num}/{total}  ", "bold white on blue"),
        ("  ", ""),
        (description, "bold cyan"),
    )
    console.print()
    console.print(Panel(text, box=box.HEAVY, border_style="blue"))
    console.print()


def print_success(message):
    console.print(f"  [bold green]✓[/bold green] {message}")


def print_info(message):
    console.print(f"  [bold blue]ℹ[/bold blue] {message}")


def print_warning(message):
    console.print(f"  [bold yellow]⚠[/bold yellow] {message}")


def print_error(message):
    console.print(f"  [bold red]✗[/bold red] {message}")


def print_kv(key, value, key_style="bold", value_style="white"):
    key_str = f"[{key_style}]{key}[/{key_style}]" if key_style else key
    val_str = f"[{value_style}]{value}[/{value_style}]" if value_style else value
    console.print(f"  [dim]│[/dim] {key_str}: {val_str}")
