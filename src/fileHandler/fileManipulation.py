from typing import Iterable, List, Tuple, Optional, Hashable
from pathlib import Path
import multiprocessing
import xarray as xr

def import_data(data_path: Iterable[Path] | Path) -> xr.Dataset:
    """import netcdf data from file(s)"""
    if isinstance(data_path, Path):
        return xr.open_dataset(data_path, engine="netcdf4")
    else:
        return read_netcdfs_concat(data_path)

def load_netcdfs(paths: Iterable[Path]) -> List[xr.Dataset]:
    """Loads a list of netCDF files"""
    return [xr.open_dataset(p.as_posix(), engine="netcdf4") for p in paths]

def read_netcdfs_concat(paths: Iterable[Path], dimension: Optional[Hashable] = None) -> xr.Dataset:
    """
    Loads all files defined in paths and concatenates them by the
    given dimension
    """
    data = load_netcdfs(paths)
    if dimension is None:
        dim_list: List[Hashable] = list(data[0].dims) # type: ignore
        dimension = dim_list[0]
    return xr.concat(data, dim=dimension)

def get_name_format(dataset: xr.Dataset) -> str:
    """Takes the first element of the datasets time and converts it to a string"""
    time = dataset.time[0].dt
    year = time.year.item()
    month = time.month.item()
    day = time.day.item()
    return f"{year}_{month}_{day}.nc"

def _export(arr: xr.Dataset, folderpath: Path):
    arr.to_netcdf(folderpath / Path(get_name_format(arr)))

def export_groups(folderpath: Path, groups: List[Tuple[int, xr.Dataset]], n_cores: int = 8) -> None:
    """
    Exports a grouped dataset.
    Each group will be exported to a file in the folderpath folder.
    The name of the each file will be the first time formatted to {year}_{month}_{day}.nc
    """

    folderpath.mkdir(parents=True, exist_ok=True)
    with multiprocessing.Pool(n_cores) as pool:
        pool.starmap(_export, [(g[1], folderpath) for g in groups])