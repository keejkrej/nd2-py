from __future__ import annotations

import os
import threading
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from .logic import export_tiff, load_nd2
from .signals import ND2Signals
from .utils import format_unit, format_value, parse_range

app = typer.Typer(help="ND2 Utilities CLI", no_args_is_help=True)
console = Console()


@app.command()
def info(
    file_path: str = typer.Argument(..., help="Path to the ND2 file"),
) -> None:
    """Display information about an ND2 file."""
    file_path = os.path.expanduser(file_path)
    if not os.path.exists(file_path):
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)

    with console.status("[bold green]Loading metadata..."):
        try:
            result: dict[str, Any] = {}
            error: list[str] = []

            def target() -> None:
                try:
                    result.update(load_nd2(file_path))
                except Exception as e:
                    error.append(str(e))

            thread = threading.Thread(target=target)
            thread.start()
            thread.join()

            if error:
                raise Exception(error[0])

            info_dict = result
        except Exception as e:
            console.print(f"[red]Error loading file: {e}[/red]")
            raise typer.Exit(1) from e

    console.print(Panel(f"[bold blue]File:[/bold blue] {file_path}", title="ND2 Info"))

    dim_table = Table(title="Dimensions")
    dim_table.add_column("Axis", style="cyan")
    dim_table.add_column("Size", style="magenta")

    dimensions = info_dict.get("dimensions", {})
    for axis, data in dimensions.items():
        dim_table.add_row(axis, str(data.get("size", 1)))

    console.print(dim_table)

    ome = info_dict.get("ome_metadata")
    if ome and ome.images:
        try:
            img0 = ome.images[0]
            px0 = img0.pixels

            summary_table = Table(title="OME Metadata Summary")
            summary_table.add_column("Property", style="cyan")
            summary_table.add_column("Value", style="magenta")

            summary_table.add_row("Pixel Type", format_value(px0.type))
            summary_table.add_row(
                "Physical Size X",
                (
                    f"{format_value(px0.physical_size_x)} "
                    f"{format_unit(px0.physical_size_x_unit)}"
                ),
            )
            summary_table.add_row(
                "Physical Size Y",
                (
                    f"{format_value(px0.physical_size_y)} "
                    f"{format_unit(px0.physical_size_y_unit)}"
                ),
            )
            if px0.physical_size_z:
                summary_table.add_row(
                    "Physical Size Z",
                    (
                        f"{format_value(px0.physical_size_z)} "
                        f"{format_unit(px0.physical_size_z_unit)}"
                    ),
                )

            if img0.acquisition_date:
                summary_table.add_row("Acquisition Date", str(img0.acquisition_date))

            summary_table.add_row("Series Count", str(len(ome.images)))
            console.print(summary_table)

            if px0.channels:
                chan_ome_table = Table(title="OME Channels")
                chan_ome_table.add_column("Index", style="cyan")
                chan_ome_table.add_column("Name", style="green")
                chan_ome_table.add_column("Color", style="magenta")
                chan_ome_table.add_column("Emission λ", style="yellow")

                for c_idx, ch in enumerate(px0.channels):
                    if hasattr(ch.color, "as_hex"):
                        color = ch.color.as_hex()
                    else:
                        color = str(ch.color)
                        if not color.startswith("#") and all(
                            c in "0123456789ABCDEFabcdef" for c in color
                        ):
                            color = f"#{color}"
                    ems = (
                        f"{format_value(ch.emission_wavelength)} "
                        f"{format_unit(ch.emission_wavelength_unit)}"
                        if ch.emission_wavelength
                        else "-"
                    )
                    chan_ome_table.add_row(
                        str(c_idx), ch.name or f"Channel {c_idx}", color, ems
                    )
                console.print(chan_ome_table)
        except Exception as e:
            console.print(
                f"[yellow]Could not display OME metadata summary: {e}[/yellow]"
            )


@app.command()
def export(
    input_path: str | None = typer.Option(
        None, "--input", "-i", help="Path to the ND2 file"
    ),
    output_path: str | None = typer.Option(
        None, "--output", "-o", help="Path to the output TIFF file"
    ),
    config: str | None = typer.Option(
        None,
        "--config",
        "-g",
        help="Path to a YAML or JSON config file for batch export",
    ),
    position: str | None = typer.Option(
        None, "--pos", "-p", help="Position range (e.g., '0-2' or '0')"
    ),
    channel: str | None = typer.Option(None, "--chan", "-c", help="Channel range"),
    time: str | None = typer.Option(None, "--time", "-t", help="Time range"),
    z: str | None = typer.Option(None, "--z", "-z", help="Z-slice range"),
) -> None:
    """Export ND2 to TIFF with selected ranges."""
    from .logic import batch_export_tiff, load_export_config

    if config:
        try:
            cfg_data = load_export_config(config)
        except Exception as e:
            console.print(f"[red]Error loading config: {e}[/red]")
            raise typer.Exit(1)

        input_file = cfg_data.get("input")
        output_dir = cfg_data.get("output_dir")
        tasks = cfg_data.get("exports", [])
        is_batch_mode = True
    else:
        if not input_path or not output_path:
            console.print(
                "[red]Error: Both --input and --output are required "
                "unless --config is used.[/red]"
            )
            raise typer.Exit(1)

        input_file = input_path
        output_dir = os.path.dirname(output_path)
        tasks = [
            {
                "output": os.path.basename(output_path),
                "pos": position,
                "chan": channel,
                "time": time,
                "z": z,
            }
        ]
        is_batch_mode = False
        cfg_data = {
            "input": input_file,
            "output_dir": output_dir,
            "exports": tasks,
        }

    if not input_file:
        console.print("[red]Error: Missing input file path.[/red]")
        raise typer.Exit(1)
    if not tasks:
        console.print("[yellow]No export tasks found.[/yellow]")
        return

    input_file = os.path.expanduser(input_file)
    if not os.path.exists(input_file):
        console.print(f"[red]Error: Input file not found: {input_file}[/red]")
        raise typer.Exit(1)

    # Display Summary
    display_text = ""
    if is_batch_mode:
        display_text += f"[bold blue]Config:[/bold blue] {config}\n"
    display_text += f"[bold blue]Input:[/bold blue] {input_file}"
    if output_dir:
        display_text += (
            f"\n[bold blue]Output Dir:[/bold blue] {os.path.expanduser(output_dir)}"
        )

    console.print(
        Panel(display_text, title="Batch Export" if is_batch_mode else "Export")
    )

    table = Table(title="Export Tasks")
    table.add_column("#", style="dim")
    table.add_column("Output", style="green")
    table.add_column("Pos", style="magenta")
    table.add_column("Chan", style="magenta")
    table.add_column("Time", style="magenta")
    table.add_column("Z", style="magenta")

    for i, t in enumerate(tasks):
        table.add_row(
            str(i + 1),
            str(t.get("output")),
            str(t.get("pos", "-") if t.get("pos") is not None else "-"),
            str(t.get("chan", "-") if t.get("chan") is not None else "-"),
            str(t.get("time", "-") if t.get("time") is not None else "-"),
            str(t.get("z", "-") if t.get("z") is not None else "-"),
        )
    console.print(table)

    if not typer.confirm("Proceed with export?", default=True):
        console.print("[yellow]Aborted.[/yellow]")
        return

    _run_export(batch_export_tiff, config=cfg_data)


def _run_export(func: Any, **kwargs: Any) -> None:
    """Helper to run export with progress reporting."""
    from .logic import export_tiff

    is_batch = func != export_tiff
    signals = ND2Signals()
    err_container: list[str] = []
    signals.error.connect(lambda msg: err_container.append(msg))

    active_progress: Progress | None = None

    def proxy_wrapper(iterable: Any, description: str | None = None) -> Any:
        import time

        if description is None:
            out = kwargs.get("output_path")
            description = (
                f"Exporting {os.path.basename(out)}..."
                if out
                else "Processing frames..."
            )

        # Wait for the UI to switch to Phase 2 (Progress Bar)
        for _ in range(100):
            if active_progress:
                break
            time.sleep(0.01)

        if active_progress:
            tid = active_progress.add_task(description, total=len(iterable))
            for item in iterable:
                yield item
                active_progress.update(tid, advance=1)
        else:
            yield from iterable

    current_progress = 0

    def update_progress(val: int) -> None:
        nonlocal current_progress
        current_progress = val

    signals.progress.connect(update_progress)

    kwargs["signals"] = signals
    kwargs["progress_wrapper"] = proxy_wrapper

    with console.status("[bold green]Loading metadata..."):
        thread = threading.Thread(target=func, kwargs=kwargs)
        thread.start()

        # Transition to progress bar UI as soon as we have any progress
        threshold = 5
        while thread.is_alive() and current_progress < threshold:
            thread.join(0.05)
            if err_container:
                break

    if err_container:
        console.print(f"[bold red]Export failed: {err_container[0]}[/bold red]")
        raise typer.Exit(1)

    if thread.is_alive():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            active_progress = progress
            # No overall progress bar as requested, only individual task bars
            signals.progress.disconnect(update_progress)

            while thread.is_alive():
                thread.join(0.05)
    else:
        pass

    if err_container:
        console.print(f"[bold red]Export failed: {err_container[0]}[/bold red]")
        raise typer.Exit(1)

    console.print("[bold green]Export complete.[/bold green]")


def main() -> None:
    """Run the ND2 CLI application."""
    app()


if __name__ == "__main__":
    main()
