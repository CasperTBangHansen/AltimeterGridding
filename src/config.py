import pydantic
import configparser
from typing import List, Tuple
import json
import re
from pathlib import Path

def is_float(element: str) -> bool:
    """ Checks if a string can be converted to a float"""
    return re.match(r'^-?\d+(?:\.\d+)$', element) is not None

def to_float(element: str) -> float:
    """ Converts string to float. Also passes fractions"""
    if is_float(element):
        return float(element)
    if "/" in element:
        frac = element.split('/')
        if len(frac) == 2 and frac[0].isnumeric() and frac[1].isnumeric():
            return float(frac[0]) / float(frac[1])            
    raise ValueError("Invalid format of blockmean_spatial_resolution")

class General(pydantic.BaseModel):
    pipeline_version: int
    multiprocessing: bool
    number_of_days: int
    overwrite_grids: bool

class Paths(pydantic.BaseModel):
    grid_path_format: str
    raw_data_path: Path
    raw_data_glob: str

class InterpolationParameters(pydantic.BaseModel):
    distance_to_time_scaling: List[float]
    n_neighbors : int
    kernel: str
    max_distance_km: float
    min_points: int

    @pydantic.validator("distance_to_time_scaling", pre=True)
    def parse_distance_to_time_scaling(cls, value: str) -> List[float]:
        return json.loads(value)

class GridParameters(pydantic.BaseModel):
    grid_resolution: float
    blockmean_spatial_resolution: float
    blockmean_temporal_resolution: int
    interpolation_groups: List[List[str]]

    @pydantic.validator("blockmean_spatial_resolution", pre=True)
    def parse_blockmean_spatial_resolution(cls, value: str) -> float:
        return to_float(value)

    @pydantic.validator("interpolation_groups", pre=True)
    def parse_interpolation_groups(cls, value: str) -> List[List[str]]:
        return json.loads(value)

def parse_config(path: Path) -> Tuple[General, GridParameters, Paths, InterpolationParameters]:
    config = configparser.ConfigParser()
    with open(path) as fd:
        config.read_file(fd)
        
    general = General(**config["General"]) # type: ignore
    gridParameters = GridParameters(**config["GridParameters"]) # type: ignore
    paths = Paths(**config["Paths"]) # type: ignore
    interpolationParameters = InterpolationParameters(**config["InterpolationParameters"]) # type: ignore
    interpolationParameters.distance_to_time_scaling = [d/general.number_of_days for d in interpolationParameters.distance_to_time_scaling]

    return general, gridParameters, paths, interpolationParameters