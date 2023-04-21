from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import xarray as xr
import numpy as np
import numpy.typing as npt
from ..fileHandler import FileMapping
from .. import config

Arguments = Tuple[
    xr.Dataset,
    FileMapping,
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    config.GridParameters,
    config.InterpolationParameters,
    str
]

@dataclass
class ConstructArguments:
    land_mask: xr.Dataset
    _file_mappings: List[FileMapping] = field(default_factory=list, init=False)
    interp_lats: npt.NDArray[np.float64]
    interp_lons: npt.NDArray[np.float64]
    gridParameters: config.GridParameters
    interpolationParameters: config.InterpolationParameters
    output_grid_path_format: str
    _pos: int = field(default=0, init=False)

    def add_file_mapping(self, file_mapping: FileMapping) -> None:
        self._file_mappings.append(file_mapping)

    def first_file(self) -> FileMapping | None:
        if self._file_mappings:
            return self._file_mappings[0]
        return None

    def current_argument(self) -> Arguments:
        return self[-1]

    def __iter__(self) -> ConstructArguments:
        self._pos = 0
        return self
    
    def __next__(self) -> Arguments:
        if self._pos < len(self._file_mappings):
            self._pos += 1
            return self[self._pos - 1]
        else:
            raise StopIteration
    
    def __getitem__(self, index: int) -> Arguments:
        return (
            self.land_mask.copy(), self._file_mappings[index], self.interp_lats,
            self.interp_lons, self.gridParameters, self.interpolationParameters,
            self.output_grid_path_format
        )

    def __bool__(self):
        return bool(self._file_mappings)