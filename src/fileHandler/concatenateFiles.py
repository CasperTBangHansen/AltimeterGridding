from tqdm import tqdm
import xarray as xr
import numpy as np
from pathlib import Path
from collections import defaultdict
import multiprocessing
from .. import correction
from .fileManipulation import load_netcdfs, read_netcdfs_concat, export_groups
from typing import List, Dict, Any, Iterable

BOUNDARY_LAT = {
    "c2": [65, -53]
}

def combine_dicts(dictionaries: Iterable[Dict[Any, Any]]) -> Dict[Any, Any | List[Any]]:
    """
    Combines the keys in the dicts by making the values lists.
    If the same key exists in multiple dicts the value will be
    added to the list with that key.
    """
    # Combine dicts by making dicts containing lists
    out_dict: Dict[str, List[Any]] = {}
    for dictionary in dictionaries:
        for attr, value in dictionary.items():
            if attr in out_dict:
                out_dict[attr].append(value)
            elif value:
                out_dict[attr] = [value]
    # If all the values of a key is the same remove the list
    # and just set the value
    for key, value in out_dict.items():
        if len(set(value)) == 1:
            out_dict[key] = value[0]
    return out_dict

def combine_attributes(datasets: Iterable[xr.Dataset], invalid_attrs: Iterable[str]) ->  Dict[Any, Any | List[Any]]:
    """
    Extracts the attributes for each dataset and combines them in a dictionary.
    Ignores attributes defined in invalid_attrs.
    """
    # Get attributes
    attrs = [dataset.attrs for dataset in datasets]
    combined_attrs = combine_dicts(attrs)

    # Drop invalid attributes
    for attr in invalid_attrs:
        combined_attrs.pop(attr, None)
    return combined_attrs

def read_netcdfs_merge(paths: Iterable[Path], dimension: str) -> xr.Dataset:
    """
    Loads all files defined in paths and concatenates them by the
    given dimension. It also combines the attributes in each dataset.
    """
    # Load data and combine it
    datasets = load_netcdfs(paths)
    combined = xr.concat(datasets, dimension)

    # Set attributes in the combined netCDF file
    combined.attrs = combine_attributes(datasets, {'log01', 'history', 'filename'})

    # Attach length attributes
    combined.attrs['n_points'] = [dataset.sizes['time'] for dataset in datasets]
    combined.attrs['total_points'] = sum(combined.attrs['n_points'])
    return combined

def group_by_name(files: Iterable[Path]) -> Dict[str, List[Path]]:
    """
    Groups paths by the file names.
    If the filenames are the same the files will be under the same key in a list.
    """
    grouped_files: Dict[str, List[Path]] = defaultdict(list)
    for file in files:
        date = file.name.replace('.nc', '')
        grouped_files[date].append(file)
    return grouped_files

def merge_with_polar(polarfiles: List[Path], data: xr.Dataset) -> xr.Dataset:
    """ Merge polar data with original dataset."""
    # Load polar data
    polar_data = (
        read_netcdfs_concat(sorted(polarfiles), 'time')
        .load()
        .set_coords(("time", "lat", "lon"))
    )
    
    # Find missing datavars
    missing_data_vars = set(data.data_vars) - set(polar_data.data_vars)

    # Set missing datavars to nan
    missing_variables = np.empty((len(missing_data_vars), polar_data.dims['time']), dtype=np.float64)
    missing_variables.fill(np.nan)
    for var, val in zip(missing_data_vars, missing_variables):
        polar_data = polar_data.assign(**{var: (['time'], val)}) # type: ignore

    # Concatenate with original
    return xr.concat([data, polar_data], dim='time').sortby('time')

def process_netcdfs(datapath: Path, polarpath: Path, crossover_basepath: Path, outputpath: Path, filematching: str) -> None:
    """Processes all the netcdf4 files into a daily file"""
    # Process data
    for satellite in (pbar := tqdm(datapath.glob("*"), position=1)):
        pbar.set_description(f"Processing {satellite.name}")
        if (outputpath / Path(satellite.name)).exists():
            continue

        # Load netcdf files
        data = read_netcdfs_concat(sorted(satellite.glob(filematching)), 'time').sortby('time').load()

        # Remove data outside boundary
        if (boundary := BOUNDARY_LAT.get(satellite.name)) is not None:
            data = data.isel(time=(data.lat <= boundary[0]) & (data.lat >= boundary[1]))

        # Get polar data of any exists
        polarfiles = list((polarpath / Path(satellite.name)).glob(filematching))
        if polarfiles:
            pbar.set_description(f"Processing {satellite.name} with polar paths")
            data = merge_with_polar(polarfiles, data)

        # Correct using crossover points
        correction.correct_netcdf(data, crossover_basepath, satellite.name, 'sla')
        
        # Group data in the netcdf files into dates
        groups = list(data.groupby('time.date'))

        # Export files
        export_groups(outputpath / Path(satellite.name), groups)

def task(folder_path: Path, date: str, group: Iterable[Path]) -> None:
    """Task for running grouping in parallel and saving them"""
    data = read_netcdfs_merge(group, 'time')
    data.to_netcdf(folder_path / Path(f"{date}.nc"))

def concatenate_date_netcdf_files(datapaths: Iterable[Path], outputpath: Path) -> None:
    """
    Concatenates all files with the same datename
    and sets some attributes in the output files
    """
    # Get files and make dirs
    outputpath.mkdir(parents=True, exist_ok=True)

    # Group files by date
    grouped_files = group_by_name(datapaths)
        
    # Process each group of files
    items = [(outputpath,) + item for item in grouped_files.items()]
    with multiprocessing.Pool(4) as pool:
        pool.starmap(task, items)