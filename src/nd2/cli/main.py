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
    input_path: str = typer.Option(..., "--input", "-i", help="Path to the ND2 file"),
    output_path: str = typer.Option(
        ..., "--output", "-o", help="Path to the output TIFF file"
    ),
    position: str | None = typer.Option(
        None, "--pos", "-p", help="Position range (e.g., '0-2' or '0')"
    ),
    channel: str | None = typer.Option(None, "--chan", "-c", help="Channel range"),
    time: str | None = typer.Option(None, "--time", "-t", help="Time range"),
    z: str | None = typer.Option(None, "--z", "-z", help="Z-slice range"),
) -> None:
    """Export ND2 to TIFF with selected ranges."""
    if not os.path.exists(input_path):
        console.print(f"[red]Error: Input file not found: {input_path}[/red]")
        raise typer.Exit(1)

    pos_range = parse_range(position)
    chan_range = parse_range(channel)
    time_range = parse_range(time)
    z_range = parse_range(z)

    console.print(f"[yellow]Exporting:[/yellow] {input_path} -> {output_path}")

    signals = ND2Signals()
    err_container: list[str] = []
    signals.error.connect(lambda msg: err_container.append(msg))

    # Proxy for the per-frame progress bar
    active_progress: Progress | None = None

    def proxy_wrapper(iterable: Any) -> Any:
        import time

        # Small wait to ensure the UI has switched to Phase 2
        for _ in range(40):
            if active_progress:
                break
            time.sleep(0.05)

        if active_progress:
            task_id = active_progress.add_task(
                "Processing frames...", total=len(iterable)
            )
            for item in iterable:
                yield item
                active_progress.update(task_id, advance=1)
        else:
            yield from iterable

    # Phase 1: Metadata Loading (Throbber)
    current_progress = 0

    def update_progress(val: int) -> None:
        nonlocal current_progress
        current_progress = val

    signals.progress.connect(update_progress)

    with console.status("[bold green]Loading metadata..."):
        thread = threading.Thread(
            target=export_tiff,
            kwargs={
                "nd2_path": input_path,
                "output_path": output_path,
                "position": pos_range,
                "channel": chan_range,
                "time": time_range,
                "z": z_range,
                "signals": signals,
                "progress_wrapper": proxy_wrapper,
            },
        )
        thread.start()

        while thread.is_alive() and current_progress < 40:
            thread.join(0.1)
            if err_container:
                break

    if err_container:
        console.print(f"[bold red]Export failed: {err_container[0]}[/bold red]")
        raise typer.Exit(1)

    # Phase 2: Data Extraction (Progress Bar)
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
            main_task = progress.add_task(
                "Exporting data...", total=100, completed=current_progress
            )
            signals.progress.disconnect(update_progress)
            signals.progress.connect(
                lambda val: progress.update(main_task, completed=val)
            )

            # Wait for completion
            while thread.is_alive():
                thread.join(0.1)
    else:
        # It might have finished very quickly
        pass

    if err_container:
        console.print(f"[bold red]Export failed: {err_container[0]}[/bold red]")
        raise typer.Exit(1)

    console.print(f"[bold green]Successfully exported to {output_path}[/bold green]")


def main() -> None:
    """Run the ND2 CLI application."""
    app()


if __name__ == "__main__":
    main()
