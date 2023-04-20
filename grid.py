import multiprocessing
import os

from typing import List, Tuple, Iterable
from pathlib import Path
from tqdm import tqdm

import xarray as xr
import numpy as np
import numpy.typing as npt
from datetime import date, timedelta
from src.interpolation import make_grid, process_grid
from src import Timer, config
from src.landmask import subset_landmask, find_masking_attributes

def file_to_date(file):
    """convert input file to date"""
    strs = file.name.split(".")[0].split("_")
    ints = list(map(int, strs))
    return date(year=ints[0], month=ints[1], day=ints[2])       

def group_valid_files(base_path: Path, files: Iterable[Path], n_days: int) -> List[Tuple[str, List[Path]]]:
    # Find all paths and get their datees
    dates = []
    for file in files:
        dt = file_to_date(file)
        dates.append(dt)
    dates.sort()
    if len(dates) != (max(dates)-min(dates)).days + 1:
        dates = [min(dates)+timedelta(days=x) for x in range((max(dates)-min(dates)).days)]
    
    # Get the file name before and after the current file,
    # but only if they are the previous/next date
    out_files = []
    for i in range(n_days, len(dates) - n_days):
        d = [dates[i]]
        for j in range(1,n_days+1):
            d.append(dates[i - j])
            d.append(dates[i + j])

        fls = [f for Date in d if (f := base_path / Path(f"{Date.year}_{Date.month}_{Date.day}.nc")).exists()]
        if fls:
            current_date_str = f"{dates[i].year}_{dates[i].month}_{dates[i].day}"
            out_files.append((current_date_str, fls))
    return out_files

def adapt_file_list(processed: Path, default_glob: str, n_days: int):

    if ((jobidx := os.environ.get("LSB_JOBINDEX")) is None):
        return group_valid_files(processed, processed.glob(default_glob), n_days=n_days)
    else:
        jobidx_int = int(jobidx)
        year_files = list(processed.glob(f"{jobidx}*.nc"))
        first_date = date(year=jobidx_int, month=1, day=1)
        last_date = date(year=jobidx_int, month=12, day=31)
        for i in range(1,n_days+1):
            date_before = (first_date - timedelta(days=i)).strftime("%Y_%m_%d")
            date_after = (last_date + timedelta(days=i)).strftime("%Y_%m_%d")
            year_files.extend([
                processed / Path(f"{date_before}.nc"),
                processed / Path(f"{date_after}.nc")
            ])
        return group_valid_files(processed, year_files, n_days=n_days)

def main():
    general, gridParameters, paths = config.parse_config(Path("config.ini"))
    # CONST
    output_grid_path_format = paths.grid_path_format.format(version=general.pipeline_version)

    timer = Timer("total")
    timer.Start()

    # Ocean mask
    land_mask_file = find_masking_attributes(gridParameters.grid_resolution, Path("ocean_mask"))
    land_mask = xr.open_dataset(land_mask_file, engine="netcdf4").load()
    land_mask = subset_landmask(land_mask, (-180, 180), (-90, 90))
    
    # Construct interpolation coordinates
    interp_lons, interp_lats = make_grid(
        gridParameters.grid_resolution,
        gridParameters.grid_resolution,
        (-180, 180),
        (-90, 90)
    )

    # Get correct glob
    # jobidx MUST BE A YEAR!
    files = adapt_file_list(processed=paths.raw_data_path, default_glob=paths.raw_data_glob, n_days=general.number_of_days)
        
    # Make commands
    commands = [
        (
            land_mask.copy(), file, interp_lats,
            interp_lons, gridParameters.interpolation_groups,
            gridParameters.blockmean_temporal_resolution, gridParameters.blockmean_spatial_resolution,
            output_grid_path_format
        )
        for file in files
    ]
    
    # Execute commands
    valid_commands: List[
        Tuple[
            xr.Dataset, Tuple[str,List[Path]], npt.NDArray[np.float64],
            npt.NDArray[np.float64], List[List[str]], int, float, str
        ]
    ] = []
    for command in tqdm(commands):
        grid_path = Path(output_grid_path_format.format(date=command[1][0]))
        if grid_path.exists():
            continue
        if general.multiprocessing:
            valid_commands.append(command)
        else:
            _ = process_grid(*command)
    if general.multiprocessing and valid_commands:
        if (command := valid_commands[0]):
            print(f"Starting from {command[1][0]}")
        with multiprocessing.Pool() as pool:
            _ = pool.starmap(process_grid, commands)
    
    print("Complete")
    timer.Stop()

if __name__ == "__main__":
    main()