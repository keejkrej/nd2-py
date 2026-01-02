from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import numpy as np

import nd2

from .signals import ND2Signals
from .utils import (
    build_slicer_dict,
    extract_data_with_progress,
    parse_dimensions,
    parse_info_dict,
    parse_range,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

logger = logging.getLogger(__name__)


def build_ome_metadata(nd2_attrs: Any, source_filename: str) -> dict[str, Any]:
    """Build OME-TIFF metadata from ND2 attributes."""
    metadata: dict[str, Any] = {
        "Description": f"Exported from ND2 file: {source_filename}"
    }
    if isinstance(nd2_attrs, dict) and "pixelSizeUm" in nd2_attrs:
        ps = nd2_attrs["pixelSizeUm"]
        for ax in ["x", "y", "z"]:
            if hasattr(ps, ax) and getattr(ps, ax):
                metadata[f"PhysicalSize{ax.upper()}"] = getattr(ps, ax)
                metadata[f"PhysicalSize{ax.upper()}Unit"] = "µm"

    if isinstance(nd2_attrs, dict) and "channelNames" in nd2_attrs:
        names = nd2_attrs["channelNames"]
        if names:
            metadata["Channel"] = {"Name": list(names)}

    return metadata


def write_tiff(output_path: str, data_5d: np.ndarray, metadata: dict[str, Any]) -> None:
    """Write 5D data to OME-TIFF."""
    from tifffile import imwrite

    t, p, c, y, x = data_5d.shape
    data_to_write = data_5d.reshape((t, p * c, y, x))
    tiff_metadata = metadata.copy()
    tiff_metadata["axes"] = "TCYX"
    imwrite(output_path, data_to_write, bigtiff=True, metadata=tiff_metadata, ome=True)


def load_nd2(
    file_path: str,
    signals: ND2Signals | None = None,
) -> dict[str, Any]:
    """Load an ND2 file and return a dictionary of information."""
    _signals = signals or ND2Signals()
    file_path = os.path.expanduser(file_path)
    try:
        with nd2.ND2File(file_path) as f:
            my_array = f.to_xarray(delayed=True)
            # metadata and attributes are accessed via properties
            info = parse_info_dict(
                file_path,
                my_array,
                {"metadata": f.metadata, "attributes": f.attributes},
            )
            info["dimensions"] = parse_dimensions(my_array)

            try:
                info["ome_metadata"] = f.ome_metadata()
            except Exception:
                logger.warning("Could not generate OME metadata")
                info["ome_metadata"] = None

            _signals.finished.emit(info)
            return info
    except Exception as e:
        logger.exception(f"Error loading ND2 file: {e}")
        _signals.error.emit(str(e))
        raise


def _export_nd2_to_tiff(
    f: nd2.ND2File,
    output_path: str,
    position: tuple[int, int] | None = None,
    channel: tuple[int, int] | None = None,
    time: tuple[int, int] | None = None,
    z: tuple[int, int] | None = None,
    signals: ND2Signals | None = None,
    progress_wrapper: Callable[[Iterable], Iterable] | None = None,
    base_progress: int = 0,
    progress_scale: float = 1.0,
) -> str:
    """Internal function to export an open ND2 file to TIFF."""
    _signals = signals or ND2Signals()

    def emit_progress(val: int) -> None:
        scaled_val = base_progress + int(val * progress_scale)
        _signals.progress.emit(scaled_val)

    emit_progress(10)
    my_array = f.to_xarray(delayed=True)
    emit_progress(20)

    nd2_attrs = f.attributes
    if hasattr(nd2_attrs, "__dict__"):
        nd2_attrs_dict = vars(nd2_attrs)
    else:
        nd2_attrs_dict = (
            dict(nd2_attrs._asdict()) if hasattr(nd2_attrs, "_asdict") else {}
        )

    emit_progress(40)
    dimensions = parse_dimensions(my_array)
    slicers = build_slicer_dict(
        position=position,
        channel=channel,
        time=time,
        z=z,
        dimensions=dimensions,
    )
    data = extract_data_with_progress(
        my_array, slicers, progress_wrapper=progress_wrapper
    )
    emit_progress(60)

    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    if data.dtype not in [np.uint8, np.uint16, np.float32]:
        if data.dtype == np.float64:
            mx = np.nanmax(data)
            data = (
                (data / mx * 65535).astype(np.uint16)
                if mx > 0
                else np.zeros_like(data, dtype=np.uint16)
            )
        else:
            data = data.astype(np.uint16)

    emit_progress(80)
    data = np.ascontiguousarray(data)
    if len(data.shape) != 5:
        # Basic 5D ensuring logic
        while len(data.shape) < 5:
            data = np.expand_dims(data, axis=0)

    metadata = build_ome_metadata(nd2_attrs_dict, os.path.basename(f.path))
    write_tiff(output_path, data, metadata)

    emit_progress(100)
    return output_path


def export_tiff(
    nd2_path: str,
    output_path: str,
    position: tuple[int, int] | None = None,
    channel: tuple[int, int] | None = None,
    time: tuple[int, int] | None = None,
    z: tuple[int, int] | None = None,
    signals: ND2Signals | None = None,
    progress_wrapper: Callable[[Iterable], Iterable] | None = None,
) -> str:
    """Export an ND2 file to a 5D OME-TIFF."""
    _signals = signals or ND2Signals()
    nd2_path = os.path.expanduser(nd2_path)
    output_path = os.path.expanduser(output_path)
    try:
        with nd2.ND2File(nd2_path) as f:
            result = _export_nd2_to_tiff(
                f,
                output_path,
                position=position,
                channel=channel,
                time=time,
                z=z,
                signals=_signals,
                progress_wrapper=progress_wrapper,
            )
            _signals.finished.emit(result)
            return result
    except Exception as e:
        logger.exception(f"Error during export: {e}")
        _signals.error.emit(str(e))
        raise


def load_export_config(config_path: str) -> dict[str, Any]:
    """Load export configuration from YAML or JSON."""
    config_path = os.path.expanduser(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    ext = os.path.splitext(config_path)[1].lower()
    if ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to load YAML configs. "
                "Install it with: pip install PyYAML"
            ) from None
        with open(config_path) as f:
            return yaml.safe_load(f)
    elif ext == ".json":
        import json

        with open(config_path) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {ext}. Use YAML or JSON.")


def batch_export_tiff(
    config: dict[str, Any],
    signals: ND2Signals | None = None,
    progress_wrapper: Callable[[Iterable, str], Iterable] | None = None,
) -> None:
    """Export multiple TIFFs from an ND2 file based on config."""
    _signals = signals or ND2Signals()
    nd2_path = config.get("input")
    if not nd2_path:
        raise ValueError("Config missing 'input' path.")

    nd2_path = os.path.expanduser(nd2_path)
    output_dir = os.path.expanduser(config.get("output_dir", ""))
    tasks = config.get("exports", [])
    if not tasks:
        _signals.finished.emit([])
        return

    try:
        with nd2.ND2File(nd2_path) as f:
            for i, task in enumerate(tasks):
                out_name = task.get("output")
                if not out_name:
                    raise ValueError(f"Export task {i} missing 'output' name.")

                out_path = os.path.expanduser(os.path.join(output_dir, out_name))
                os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

                # Wrap progress_wrapper if provided to include task description
                task_wrapper: Any = None
                if progress_wrapper is not None:
                    pw = progress_wrapper
                    name = out_name
                    task_wrapper = lambda it: pw(it, f"Exporting {name}...")

                # Scale progress for each task: (i/len)*100 to ((i+1)/len)*100
                base = int((i / len(tasks)) * 100)
                scale = 1.0 / len(tasks)

                _export_nd2_to_tiff(
                    f,
                    out_path,
                    position=parse_range(task.get("pos")),
                    channel=parse_range(task.get("chan")),
                    time=parse_range(task.get("time")),
                    z=parse_range(task.get("z")),
                    signals=_signals,
                    progress_wrapper=task_wrapper,
                    base_progress=base,
                    progress_scale=scale,
                )

            _signals.progress.emit(100)
            _signals.finished.emit([t.get("output") for t in tasks])
    except Exception as e:
        logger.exception(f"Error during batch export: {e}")
        _signals.error.emit(str(e))
        raise
