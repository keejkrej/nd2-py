from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Callable

logger = logging.getLogger(__name__)


STANDARD_AXES: list[str] = ["T", "P", "C", "Y", "X"]


def parse_dimensions(xarray: Any) -> dict[str, dict[str, Any]]:
    """Parse dimension information from xarray."""
    dimension_info = {}
    for axis in xarray.dims:
        dimension_info[axis] = {
            "size": xarray.sizes.get(axis, 1),
            "labels": [],
        }
    return dimension_info


def validate_dimension_selection(
    dimensions: dict[str, dict[str, Any]],
    position: int | None = None,
    channel: int | None = None,
    time: int | None = None,
    z: int | None = None,
) -> dict[str, int]:
    """Validate and return valid dimension selections."""
    slicers = {}

    if position is not None and "P" in dimensions:
        if 0 <= position < dimensions["P"]["size"]:
            slicers["P"] = position

    if channel is not None and "C" in dimensions:
        if 0 <= channel < dimensions["C"]["size"]:
            slicers["C"] = channel

    if time is not None and "T" in dimensions:
        if 0 <= time < dimensions["T"]["size"]:
            slicers["T"] = time

    if z is not None and "Z" in dimensions:
        if 0 <= z < dimensions["Z"]["size"]:
            slicers["Z"] = z

    return slicers


def extract_data_with_progress(
    xarray: Any,
    slicers: dict[str, Any],
    desc: str = "Processing data",
    progress_wrapper: Callable[[Iterable], Iterable] | None = None,
) -> np.ndarray:
    """Extract data in batches using isel wrapper with progress bar."""
    dim_indices = {}
    batch_dims = []
    fixed_dims = {}

    for dim, value in slicers.items():
        if dim not in xarray.dims:
            continue

        if isinstance(value, (list, tuple)) and len(value) == 2:
            start, end = value
            if start == end:
                fixed_dims[dim] = start
            else:
                indices = list(range(start, end + 1))
                dim_indices[dim] = indices
                if len(indices) > 1:
                    batch_dims.append(dim)
        elif isinstance(value, (list, tuple)) and len(value) > 2:
            dim_indices[dim] = list(value)
            batch_dims.append(dim)
        elif value is not None:
            fixed_dims[dim] = value

    if not batch_dims:
        result = xarray.isel(fixed_dims, drop=False).compute()
        if hasattr(result, "dtype") and result.dtype != xarray.dtype:
            result = result.astype(xarray.dtype)
        return np.asarray(result)

    batch_combinations = list(
        itertools.product(*[dim_indices[dim] for dim in batch_dims])
    )

    final_shape = []
    for dim in xarray.dims:
        if dim in batch_dims:
            final_shape.append(len(dim_indices[dim]))
        elif dim in fixed_dims:
            final_shape.append(1)
        else:
            final_shape.append(xarray.sizes[dim])

    result_array = np.zeros(final_shape, dtype=xarray.dtype)
    index_maps = {
        dim: {
            orig_idx: result_idx for result_idx, orig_idx in enumerate(dim_indices[dim])
        }
        for dim in batch_dims
    }
    dim_positions = {dim: i for i, dim in enumerate(xarray.dims)}

    iterator = (
        progress_wrapper(batch_combinations) if progress_wrapper else batch_combinations
    )

    for combination in iterator:
        current_slicers = fixed_dims.copy()
        for i, dim in enumerate(batch_dims):
            current_slicers[dim] = combination[i]

        chunk = xarray.isel(current_slicers, drop=False).compute()
        if hasattr(chunk, "dtype") and chunk.dtype != xarray.dtype:
            chunk = chunk.astype(xarray.dtype)

        array_indices: list[Any] = [slice(None)] * len(xarray.dims)
        for i, dim in enumerate(batch_dims):
            pos = dim_positions[dim]
            array_indices[pos] = index_maps[dim][combination[i]]

        for dim in fixed_dims:
            pos = dim_positions[dim]
            array_indices[pos] = 0

        result_array[tuple(array_indices)] = chunk

    return result_array


def build_slicer_dict(
    position: Any = None,
    channel: Any = None,
    time: Any = None,
    z: Any = None,
    dimensions: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build slicer dictionary from individual dimension selections."""
    slicers = {}
    if dimensions is None:
        if position is not None:
            slicers["P"] = position
        if channel is not None:
            slicers["C"] = channel
        if time is not None:
            slicers["T"] = time
        if z is not None:
            slicers["Z"] = z
        return slicers

    for key, dim_key in [("P", position), ("C", channel), ("T", time), ("Z", z)]:
        if key in dimensions:
            slicers[key] = (
                (0, dimensions[key]["size"] - 1) if dim_key is None else dim_key
            )

    return slicers


def convert_attrs_to_dict(attrs_obj: Any) -> dict[str, Any]:
    """Convert ND2 attributes object to dictionary."""
    if hasattr(attrs_obj, "__dict__"):
        return vars(attrs_obj)
    if hasattr(attrs_obj, "_asdict"):
        return attrs_obj._asdict()
    try:
        return dict(attrs_obj) if attrs_obj else {}
    except (TypeError, ValueError):
        return {"raw_object": str(attrs_obj)}


def extract_pixel_size(attrs_obj: Any) -> dict[str, Any]:
    """Extract pixel size information from ND2 attributes."""
    if not hasattr(attrs_obj, "pixelSizeUm"):
        return {}
    ps = attrs_obj.pixelSizeUm
    if hasattr(ps, "__dict__"):
        d = vars(ps)
    elif hasattr(ps, "_asdict"):
        d = ps._asdict()
    else:
        d = {
            "x": getattr(ps, "x", None),
            "y": getattr(ps, "y", None),
            "z": getattr(ps, "z", None),
        }
    return {k: v for k, v in d.items() if v is not None}


def parse_info_dict(
    file_path: str, xarray: Any, attrs: dict[str, Any]
) -> dict[str, Any]:
    """Build comprehensive information dictionary from ND2 data."""
    nd2_attrs = attrs.get("attributes", {})
    return {
        "path": file_path,
        "shape": xarray.shape,
        "size": xarray.size,
        "dtype": str(xarray.dtype),
        "axes": list(xarray.dims),
        "metadata": attrs.get("metadata", {}),
        "attributes": convert_attrs_to_dict(nd2_attrs),
        "xarray": xarray,
        "pixel_size": extract_pixel_size(nd2_attrs),
    }


def parse_range(range_val: str | int | None) -> tuple[int, int] | None:
    """Parse a range like '0-2', '1', or 1 into a tuple (start, end)."""
    if range_val is None:
        return None
    if isinstance(range_val, int):
        return (range_val, range_val)
    if not isinstance(range_val, str):
        return None

    range_str = range_val.strip()
    if not range_str:
        return None
    if "-" in range_str:
        parts = range_str.split("-")
        if len(parts) == 2:
            try:
                return (int(parts[0]), int(parts[1]))
            except ValueError:
                return None
    try:
        val = int(range_str)
        return (val, val)
    except ValueError:
        return None


def format_value(val: Any) -> str:
    """Format numeric values for display."""
    if hasattr(val, "value"):
        val = val.value
    if isinstance(val, float):
        return f"{val:.4f}".rstrip("0").rstrip(".")
    return str(val)


def format_unit(unit: Any) -> str:
    """Format physical units for display."""
    s = str(unit).split(".")[-1].lower()
    mapping = {
        "micrometer": "µm",
        "nanometer": "nm",
        "millisecond": "ms",
        "second": "s",
    }
    return mapping.get(s, s)
