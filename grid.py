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

def group_valid_files(base_path: Path, files: Iterable[Path]) -> List[List[Path]]:
    # Find all paths and get their datees
    dates = []
    for file in files:
        dt = file_to_date(file)
        dates.append(dt)
    dates.sort()
    
    # Get the file name before and after the current file,
    # but only if they are the previous/next date
    out_files = []
    for i in range(1, len(dates) - 1):
        d = []
        if dates[i] - timedelta(days=1) == dates[i-1]:
            d.append(dates[i-1])
        d.append(dates[i])
        if dates[i] + timedelta(days=1) == dates[i+1]:
            d.append(dates[i + 1])

        fls = [base_path / Path(f"{Date.year}_{Date.month}_{Date.day}.nc") for Date in d]
        out_files.append(fls)
    return out_files

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
    # jobidx MOST BE A YEAR!
    if ((jobidx := os.environ.get("LSB_JOBINDEX")) is None):
        files = group_valid_files(paths.raw_data_path, paths.raw_data_path.glob(paths.raw_data_glob))
    else:
        jobidx_int = int(jobidx)
        year_files = list(paths.raw_data_path.glob(f"{jobidx}*.nc"))
        year_files.extend([
            paths.raw_data_path / Path(f"{jobidx_int - 1}_12_31.nc"),
            paths.raw_data_path / Path(f"{jobidx_int + 1}_1_1.nc")
        ])
        files = group_valid_files(paths.raw_data_path, year_files)
        
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
            xr.Dataset, List[Path], npt.NDArray[np.float64],
            npt.NDArray[np.float64], List[List[str]], float, float, str
        ]
    ] = []
    for command in tqdm(commands):
        date_str = command[1][1].name.split('.')[0]
        grid_path = Path(output_grid_path_format.format(date=date_str))
        if grid_path.exists():
            continue
        if general.multiprocessing:
            valid_commands.append(command)
        else:
            _ = process_grid(*command)
    if general.multiprocessing and valid_commands:
        if (command := valid_commands[0]):
            date_str = command[1][1].name.split('.')[0]
            print(f"Starting from {date_str}")
        with multiprocessing.Pool() as pool:
            _ = pool.starmap(process_grid, commands)
    
    print("Complete")
    timer.Stop()

if __name__ == "__main__":
    main()