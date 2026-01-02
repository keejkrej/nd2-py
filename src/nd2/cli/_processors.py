from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import nd2
import numpy as np

from ._core import DefaultSignals, OperationCancelled
from ._utils import DimensionParser, MetadataHandler

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from ._core import SignalsInterface

logger = logging.getLogger(__name__)


def build_ome_metadata(nd2_attrs: Any, source_filename: str) -> dict[str, Any]:
    """Build OME-TIFF metadata from ND2 attributes."""
    metadata = {"Description": f"Exported from ND2 file: {source_filename}"}
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


class ND2ProcessorLogic:
    @staticmethod
    def load(
        file_path: str,
        signals: SignalsInterface | None = None,
    ) -> dict[str, Any]:
        _signals: SignalsInterface = signals or DefaultSignals()  # type: ignore
        try:
            with nd2.ND2File(file_path) as f:
                my_array = f.to_xarray(delayed=True)
                # metadata and attributes are accessed via properties
                info = MetadataHandler.build_info_dict(
                    file_path,
                    my_array,
                    {"metadata": f.metadata, "attributes": f.attributes},
                )
                info["dimensions"] = DimensionParser.parse_dimensions(my_array)

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


class TiffExportLogic:
    @staticmethod
    def export(
        nd2_path: str,
        output_path: str,
        position: tuple[int, int] | None = None,
        channel: tuple[int, int] | None = None,
        time: tuple[int, int] | None = None,
        z: tuple[int, int] | None = None,
        signals: SignalsInterface | None = None,
        progress_wrapper: Callable[[Iterable], Iterable] | None = None,
    ) -> str:
        _signals: SignalsInterface = signals or DefaultSignals()  # type: ignore
        try:
            _signals.progress.emit(10)
            with nd2.ND2File(nd2_path) as f:
                my_array = f.to_xarray(delayed=True)
                _signals.progress.emit(20)

                nd2_attrs = f.attributes
                if hasattr(nd2_attrs, "__dict__"):
                    nd2_attrs_dict = vars(nd2_attrs)
                else:
                    nd2_attrs_dict = (
                        dict(nd2_attrs._asdict())
                        if hasattr(nd2_attrs, "_asdict")
                        else {}
                    )

                _signals.progress.emit(40)
                dimensions = DimensionParser.parse_dimensions(my_array)
                slicers = DimensionParser.build_slicer_dict(
                    position=position,
                    channel=channel,
                    time=time,
                    z=z,
                    dimensions=dimensions,
                )
                data = DimensionParser.extract_data_with_progress(
                    my_array, slicers, progress_wrapper=progress_wrapper
                )
                _signals.progress.emit(60)

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

                _signals.progress.emit(80)
                data = np.ascontiguousarray(data)
                if len(data.shape) != 5:
                    # Basic 5D ensuring logic
                    while len(data.shape) < 5:
                        data = np.expand_dims(data, axis=0)

                metadata = build_ome_metadata(
                    nd2_attrs_dict, os.path.basename(nd2_path)
                )
                write_tiff(output_path, data, metadata)

                _signals.progress.emit(100)
                _signals.finished.emit(output_path)
                return output_path
        except Exception as e:
            logger.exception(f"Error during export: {e}")
            _signals.error.emit(str(e))
            raise
