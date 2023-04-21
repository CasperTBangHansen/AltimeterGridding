from typing import Iterable, List, Tuple, Optional, Hashable, NamedTuple
from pathlib import Path
import os
from datetime import date, timedelta
import multiprocessing
import xarray as xr

class FileMapping(NamedTuple):
    computation_date: date
    files: List[Path]

    @property
    def computation_date_str(self) -> str:
        return f"{self.computation_date.year}_{self.computation_date.month}_{self.computation_date.day}"

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
    

def file_to_date(file):
    """convert input file to date"""
    strs = file.name.split(".")[0].split("_")
    ints = list(map(int, strs))
    return date(year=ints[0], month=ints[1], day=ints[2])       

def group_valid_files(base_path: Path, files: Iterable[Path], n_days: int) -> List[FileMapping]:
    # Find all paths and get their datees
    dates: List[date] = []
    for file in files:
        dt = file_to_date(file)
        dates.append(dt)
    dates.sort()
    # if len(dates) != (max(dates) - min(dates)).days + 1:
    min_date = min(dates) 
    date_span = (min_date - min(dates)).days + 2*n_days + 1
    dates = [min_date + timedelta(days=x - n_days) for x in range(date_span)]
    
    # Get the file name before and after the current file,
    # but only if they are the previous/next date
    out_files = []
    for i in range(n_days, len(dates) - n_days):
        d = [dates[i]]
        for j in range(1, n_days + 1):
            d.append(dates[i - j])
            d.append(dates[i + j])

        fls = [f for Date in d if (f := base_path / Path(f"{Date.year}_{Date.month}_{Date.day}.nc")).exists()]
        if fls:
            out_files.append(FileMapping(dates[i], fls))
    return out_files

def locate_date_source(processed: Path, default_glob: str, n_days: int) -> List[FileMapping]:

    if ((jobidx := os.environ.get("LSB_JOBINDEX")) is None or jobidx == '0'):
        year_files = processed.glob(default_glob)
    else:
        year_files = processed.glob(f"{jobidx}*.nc")
    return group_valid_files(processed, year_files, n_days=n_days)
