import xarray as xr
import numpy as np
from .processing import group_by_time
from pathlib import Path
from typing import Tuple, Dict, List, NamedTuple

class SatelliteIndexs(NamedTuple):
    satellite: int
    reference: int

MAIN_REFERENCE = "tx"
IS_REFERENCE: Tuple[str, ...] = ("tx" ,"j1", "j2", "j3")

CROSSOVER_REFERENCE: Dict[str, Tuple[str,...]] = {
    'e1': ('tx',),
    'e2': ('tx', 'j1'),
    'j1': ('tx', ),
    'n1': ('tx', 'j1'),
    'j2': ('j1',),
    'c2': ('j1', 'j2', 'j3'),
    'sa': ('j1', 'j2', 'j3'),
    'j3': ('j2',),
    '3a': ('j2', 'j3'),
    '3b': ('j2', 'j3'),
    '6a': ('j3',)
}

def correct_crossovers(crossover: xr.Dataset, crossover_indexes: SatelliteIndexs, reference_crossover: xr.Dataset, reference_crossover_indexes: SatelliteIndexs, datavar: str):
    """
    Correct crossover to reference crossovers. This both crossover and reference_crossover should containing the same satellite.
    Example:
        Crossover: [j1, e2]
        reference_crossover: [tx, j1]
        crossover_indexes: (1, 0)
        reference_crossover_indexes: (1, 0)
    """
    crossover = crossover.sortby(crossover.time[:, crossover_indexes.reference])

    # Threshold time for crossover
    greater_than_crossover = crossover.time.isel(leg=crossover_indexes.reference) >= reference_crossover.time.isel(xover=0, leg=reference_crossover_indexes.satellite)
    less_than_crossover = crossover.time.isel(leg=crossover_indexes.reference) <= reference_crossover.time.isel(xover=-1, leg=reference_crossover_indexes.satellite)
    # Select data
    valid_crossovers = crossover.isel(xover = greater_than_crossover & less_than_crossover)
    
    # Threshold time for reference crossover
    #greater_than_ref_crossover = reference_crossover.time.isel(leg=reference_crossover_indexes.satellite) >= valid_crossovers.time.isel(xover=0, leg=crossover_indexes.reference)
    #less_than_ref_crossover = reference_crossover.time.isel(leg=reference_crossover_indexes.satellite) <= valid_crossovers.time.isel(xover=-1, leg=crossover_indexes.reference)
    # Select data
    #valid_ref_crossovers = reference_crossover.isel(xover = greater_than_ref_crossover & less_than_ref_crossover)

    # Get difference in datavar at each crossover
    crossover_ref = reference_crossover[[datavar, 'time']]
    diff = crossover_ref[datavar][:, reference_crossover_indexes.reference] - crossover_ref[datavar][:, reference_crossover_indexes.satellite]
    diff_between = diff[1:] - diff[:-1]

    # Crossover time
    start_time = crossover_ref['time'][:-1, reference_crossover_indexes.satellite].astype(np.int64)
    end_time = crossover_ref['time'][1:, reference_crossover_indexes.satellite].astype(np.int64)
    time_diff = end_time - start_time

    # Compute slope
    slopes = diff_between/time_diff

    # Group data between each pair of crossover points
    times = valid_crossovers.time[:, crossover_indexes.reference].data.astype(np.int64)
    crossover_time = crossover_ref['time'][:, crossover_indexes.satellite].data.astype(np.int64)
    indexing = group_by_time(times, crossover_time)

    # Correct data
    correction = slopes[indexing - 1] * (times - crossover_time[indexing - 1])
    correction[indexing == 1] -= diff_between[0]
    crossover[datavar][greater_than_crossover & less_than_crossover, crossover_indexes.reference] = (
        valid_crossovers[datavar][:, crossover_indexes.reference] + correction.data
    )
    
    # Sort by satellite time
    return crossover.sortby(crossover.time[:, crossover_indexes.satellite])

def _load_crossover(crossover_path: Path, crossover_sat: str, satellite_name: str, datavar: str) -> Tuple[xr.Dataset, SatelliteIndexs]:
    """
    Loads a crossover netcdf file based on the path and the names of the satellites.
    The time parameter and the datavar parameter is the only two parameters which will be loaded.
    The time will be returned sorted.
    The satellite index is the index in the dataset which will map to the current satellite and
    the reference index is the index  in the dataset which will map to the reference satellite and
    """
    crossover_all = xr.open_dataset((crossover_path / Path(f"{crossover_sat}_{satellite_name}.nc")).as_posix())
    crossover = crossover_all[[datavar, 'time']].load()
    satellite_names = crossover_all.legs.split(' ')
    sat_idx = satellite_names.index(satellite_name)
    ref_idx = satellite_names.index(crossover_sat)
    crossover = crossover.sortby(crossover.time[:, sat_idx])
    return crossover, SatelliteIndexs(sat_idx, ref_idx)

def load_crossover(crossover_sats: Tuple[str,...], crossover_path: Path, satellite_name: str, datavar: str = 'sla') -> Tuple[xr.Dataset, SatelliteIndexs]:
    """
    Loads crossover files for a given satellite and corrects the crossover points for each pair of sattelites 
    using the previous pairs of crossover point
    """
    crossovers: List[xr.Dataset] = []
    satellite_indexses: List[SatelliteIndexs] = []
    for i, crossover_sat in enumerate(crossover_sats):
        # Load crossover satellite mapped to current satellite
        crossover, _satellite_indexses = _load_crossover(crossover_path, crossover_sat, satellite_name, datavar)
        satellite_indexses.append(_satellite_indexses)

        # Correct crossover satellite with the previous crossover satellite
        if i > 0:
            crossover_corr, cor_satellite_indexs = _load_crossover(crossover_path, crossover_sats[i - 1], crossover_sat, datavar)
            crossover = correct_crossovers(crossover, satellite_indexses[-1], crossover_corr, cor_satellite_indexs, datavar)
            
            # Only select crossovers which are later than the last reference satellites crossovers
            # new_time = crossover.time.isel(leg=satellite_indexses[-1].satellite)
            # prev_time = crossovers[-1].time.isel(xover=-1, leg=satellite_indexses[-2].satellite)
            # crossover = crossover.isel(xover = new_time >= prev_time)
            
            # Switch crossover to the new crossover whenever it begins
            new_time = crossover.time.isel(xover=0, leg=satellite_indexses[-1].satellite)
            prev_time = crossovers[-1].time.isel(leg=satellite_indexses[-2].satellite)
            crossovers[-1] = crossovers[-1].isel(xover = new_time >= prev_time)
        crossovers.append(crossover)
    
    # Concat
    if len(satellite_indexses) == 0:
        raise ValueError("crossover_sats did not containing any satellites")

    crossover = xr.concat(crossovers, dim='xover')
    crossover = crossover.sortby(crossover.time[:, satellite_indexses[0].satellite])
    return crossover, satellite_indexses[0]

def correct_netcdf(data: xr.Dataset, crossover_path: Path, satellite_name: str, datavar: str = 'sla'):
    """Corrects the data based on crossover points. The input data will be updated inplace."""
    # Dont correct if the satellite is the main reference satellite
    # Or no crossover points exists
    if satellite_name == MAIN_REFERENCE or (crossover_sats := CROSSOVER_REFERENCE.get(satellite_name)) is None:
        return None
    
    # Get crossover satellites
    crossovers, satellite_indexes = load_crossover(crossover_sats, crossover_path, satellite_name, datavar)

    # Get data within the crossover points time frame
    index_interval = get_valid_indexes(data, crossovers, satellite_indexes.satellite)

    # Correct the data using the crossover points
    data['sla'][index_interval] = correct_data(data.isel(time=index_interval), crossovers, datavar, satellite_indexes)

def get_valid_indexes(sat: xr.Dataset, crossover: xr.Dataset, sat_idx: int) -> slice:
    """Get indexes which maps to the time which are within the crossover time interval"""
    # Threshold time
    greater_than = sat.time >= crossover.time.isel(xover=0, leg=sat_idx)
    less_than = sat.time <= crossover.time.isel(xover=-1, leg=sat_idx)
    
    # Get amount of data remove before and after the threshold
    start_idx = (~greater_than).data.sum()
    end_idx = len(sat['time']) - (~less_than).data.sum()

    # Select data
    return slice(start_idx, end_idx)

def correct_data(sat: xr.Dataset, crossover: xr.Dataset, datavar: str, indexes: SatelliteIndexs):
    """ Corrects the satellite data based on the crossover points and the reference satellite"""
    # Get difference in datavar at each crossover
    crossover = crossover[[datavar, 'time']]
    diff = crossover[datavar][:, indexes.reference] - crossover[datavar][:, indexes.satellite]
    diff_between = diff[1:] - diff[:-1]

    # Crossover time
    start_time = crossover['time'][:-1, indexes.satellite].astype(np.int64)
    end_time = crossover['time'][1:, indexes.satellite].astype(np.int64)
    time_diff = end_time - start_time

    # Remove when time has been to long
    invalid_times = ((time_diff*10**(-9))/(60*60) > 24)

    # Compute slope
    slopes = diff_between/time_diff
    slopes[invalid_times] = 0


    # Group data between each pair of crossover points
    times = sat.time.data.astype(np.int64)
    crossover_time = crossover['time'][:, indexes.satellite].values.astype(np.int64)
    indexing = group_by_time(times, crossover_time)

    # Correct data
    correction = slopes[indexing - 1] * (times - crossover_time[indexing - 1]) + (slopes[indexing - 1] != 0) * diff[indexing - 1]
    return sat[datavar] + correction.data