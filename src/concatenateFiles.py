from tqdm import tqdm
import xarray as xr
from pathlib import Path
from collections import defaultdict
import multiprocessing
from typing import List, Tuple, Dict, Any, Iterable

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

def load_netcdfs(paths: Iterable[Path]) -> List[xr.Dataset]:
    """Loads a list of netCDF files"""
    return [xr.open_dataset(p.as_posix()) for p in paths]

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

def read_netcdfs_concat(paths: Iterable[Path], dimension: str) -> xr.Dataset:
    """
    Loads all files defined in paths and concatenates them by the
    given dimension
    """
    return xr.concat(load_netcdfs(paths), dimension)

def get_name_format(dataset: xr.Dataset) -> str:
    """Takes the first element of the datasets time and converts it to a string"""
    time = dataset.time[0].dt
    year = time.year.item()
    month = time.month.item()
    day = time.day.item()
    return f"{year}_{month}_{day}.nc"

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

def export_groups(folderpath: Path, groups: List[Tuple[int, xr.Dataset]]) -> None:
    """
    Exports a grouped dataset.
    Each group will be exported to a file in the folderpath folder.
    The name of the each file will be the first time formatted to {year}_{month}_{day}.nc
    """
    folderpath.mkdir(parents=True, exist_ok=True)
    for _, arr in groups:
        arr.to_netcdf(folderpath / Path(get_name_format(arr)))

def process_netcdfs(datapath: Path, outputpath: Path, filematching: str) -> None:
    """Processes all the netcdf4 files into a daily file"""
    # Process data
    for satellite in (pbar := tqdm(datapath.glob("*"), position=1)):
        pbar.set_description(f"Processing {satellite.name}")
        
        # Load netcdf files
        data = read_netcdfs_concat(sorted(satellite.glob(filematching)), 'time')

        # Group data in the netcdf files into dates
        groups = list(data.groupby('time.date'))

        # Export files
        export_groups(outputpath / Path(satellite.name), groups)

def main_cycle_to_dates() -> None:
    """
    Converts netcdf files containing cycles to
    netcdf files containing dates
    """
    DATAPATH = Path("radsCycles")
    PROCESSEDPATH = Path("Processed")
    FILEMATCHING = '*.nc'
    process_netcdfs(DATAPATH, PROCESSEDPATH, FILEMATCHING)

def task(folder_path: Path, date: str, group: Iterable[Path]) -> None:
    """Task for running grouping in parallel and saving them"""
    data = read_netcdfs_merge(group, 'time')
    data.to_netcdf(folder_path / Path(f"{date}.nc"))

def concatenate_date_netcdf_files() -> None:
    """
    Concatenates all files with the same datename
    and sets some attributes in the output files
    """
    DATAPATH = Path("Processed")
    OUTFOLDER = DATAPATH / Path('all')

    # Get files and make dirs
    files = DATAPATH.glob('[!all]*/*.nc')
    OUTFOLDER.mkdir(parents=True, exist_ok=True)

    # Group files by date
    grouped_files = group_by_name(files)
        
    # Process each group of files
    items = [(OUTFOLDER,) + item for item in grouped_files.items()]
    with multiprocessing.Pool(4) as pool:
        pool.starmap(task, items)

if __name__ == '__main__':
    main_cycle_to_dates()
    concatenate_date_netcdf_files()