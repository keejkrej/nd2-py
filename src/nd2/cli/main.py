from __future__ import annotations

import os
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

from ._core import SignalsInterface, WorkerSignal
from ._processors import ND2ProcessorLogic, TiffExportLogic

app = typer.Typer(help="ND2 Utilities CLI", no_args_is_help=True)
console = Console()


class RichSignal(WorkerSignal):
    def emit(self, *args: Any) -> None:
        pass


class RichSignals(SignalsInterface):
    def __init__(self) -> None:
        self.progress = RichSignal()
        self.finished = RichSignal()
        self.error = RichSignal()


def parse_range(range_str: str | None) -> tuple[int, int] | None:
    if not range_str:
        return None
    if "-" in range_str:
        parts = range_str.split("-")
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
    try:
        val = int(range_str)
        return (val, val)
    except ValueError:
        return None


def _fmt_val(val: Any) -> str:
    if hasattr(val, "value"):
        val = val.value
    if isinstance(val, float):
        return f"{val:.4f}".rstrip("0").rstrip(".")
    return str(val)


def _fmt_unit(unit: Any) -> str:
    s = str(unit).split(".")[-1].lower()
    mapping = {
        "micrometer": "µm",
        "nanometer": "nm",
        "millisecond": "ms",
        "second": "s",
    }
    return mapping.get(s, s)


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
            info_dict = ND2ProcessorLogic.load(file_path)
        except Exception as e:
            console.print(f"[red]Error loading file: {e}[/red]")
            raise typer.Exit(1)

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
            # Show summary from the first image
            img0 = ome.images[0]
            px0 = img0.pixels

            summary_table = Table(title="OME Metadata Summary")
            summary_table.add_column("Property", style="cyan")
            summary_table.add_column("Value", style="magenta")

            summary_table.add_row("Pixel Type", _fmt_val(px0.type))
            summary_table.add_row(
                "Physical Size X",
                f"{_fmt_val(px0.physical_size_x)} {_fmt_unit(px0.physical_size_x_unit)}",
            )
            summary_table.add_row(
                "Physical Size Y",
                f"{_fmt_val(px0.physical_size_y)} {_fmt_unit(px0.physical_size_y_unit)}",
            )
            if px0.physical_size_z:
                summary_table.add_row(
                    "Physical Size Z",
                    f"{_fmt_val(px0.physical_size_z)} {_fmt_unit(px0.physical_size_z_unit)}",
                )

            if img0.acquisition_date:
                summary_table.add_row("Acquisition Date", str(img0.acquisition_date))

            summary_table.add_row("Series Count", str(len(ome.images)))
            console.print(summary_table)

            # Channel info from the first image
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
                        f"{_fmt_val(ch.emission_wavelength)} {_fmt_unit(ch.emission_wavelength_unit)}"
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

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        main_task = progress.add_task("Preparing export...", total=100)

        def signals_progress_emit(val: int) -> None:
            progress.update(main_task, completed=val)

        class Signals:
            def __init__(self) -> None:
                self.progress = type("obj", (object,), {"emit": signals_progress_emit})
                self.finished = type("obj", (object,), {"emit": lambda x: None})
                self.error = type(
                    "obj",
                    (object,),
                    {"emit": lambda x: console.print(f"[red]{x}[/red]")},
                )

        def rich_wrapper(iterable: Any) -> Any:
            return progress.track(iterable, description="Extracting data...")

        try:
            TiffExportLogic.export(
                nd2_path=input_path,
                output_path=output_path,
                position=pos_range,
                channel=chan_range,
                time=time_range,
                z=z_range,
                signals=Signals(),  # type: ignore
                progress_wrapper=rich_wrapper,
            )
            console.print(
                f"[bold green]Successfully exported to {output_path}[/bold green]"
            )
        except Exception as e:
            console.print(f"[bold red]Export failed: {e}[/bold red]")
            raise typer.Exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
