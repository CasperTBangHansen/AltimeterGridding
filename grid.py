import subprocess
import multiprocessing
import os
import time

from typing import List, Tuple
from pathlib import Path
from tqdm import tqdm

import xarray as xr
import numpy as np
import numpy.typing as npt
from datetime import date, timedelta

# from scipy.interpolate import RBFInterpolator
from grid_funcs import sign_add, _landmask_coord_bool, block_mean_loop_time
from src import RBFInterpolator


def find_masking_attributes(resolution_deg: float) -> str:
    """Determine land masking from resolution"""
    base_path = Path("data")
    if not base_path.exists():
        base_path.mkdir()
    if resolution_deg == 1:
        land_mask_file = Path("land_NaN_01d.grd")
        mask_name = "earth_mask_01d_p"
    elif resolution_deg == 1/4:
        land_mask_file = Path("land_NaN_15m.grd")
        mask_name = "earth_mask_15m_p"
    elif resolution_deg == 1/6:
        land_mask_file = Path("land_NaN_10m.grd")
        mask_name = "earth_mask_10m_p"
    elif resolution_deg == 1/12:
        land_mask_file = Path("land_NaN_05m.grd")
        mask_name = "earth_mask_05m_p"
    else:
        raise ValueError("Invalid grid resolution. Valid resolutions are 1, 1/4, 1/6 or 1/12 degrees.")
    
    land_mask_file = base_path / land_mask_file
    if not land_mask_file.is_file():
        command = f"gmt grdmath @{mask_name} 0 LE 0 NAN = {land_mask_file}"
        subprocess.run(command, stdout=open(os.devnull, 'wb'))
    return land_mask_file.as_posix()

def make_grid(x_deg: float, y_deg: float, x_boundary: Tuple[float, float], y_boundary: Tuple[float, float]) -> List[npt.NDArray[np.float64]]:
    """Creates a grid of x, y"""
    x_start = sign_add(x_boundary[0], x_deg/2)
    x_end = sign_add(x_boundary[1], -x_deg/2)
    x = np.arange(x_start, x_end, x_deg)
    y_start = sign_add(y_boundary[0], y_deg/2)
    y_end = sign_add(y_boundary[1], -y_deg/2)
    y = np.arange(y_start, y_end, y_deg)
    return np.meshgrid(x, y)

def subset_landmask(landmask: xr.Dataset, x_boundary: Tuple[float, float], y_boundary: Tuple[float, float]) -> xr.Dataset:
    """ Takes a subset of the landmask based on the x and y boundaries"""
    lat_min = _landmask_coord_bool(landmask.lat.values, y_boundary[0])
    lat_max = _landmask_coord_bool(landmask.lat.values, y_boundary[1])
    lon_min = _landmask_coord_bool(landmask.lon.values, x_boundary[0])
    lon_max = _landmask_coord_bool(landmask.lon.values, x_boundary[1])
    return landmask.isel(lat=slice(lat_min,lat_max), lon=slice(lon_min,lon_max))

def block_mean(x_boundary: Tuple[float, float],y_boundary: Tuple[float, float], data_path: List[Path] | Path) -> npt.NDArray[np.float64]:
    """mean grid in blocks of resolution size"""
    # timer = Timer("Block mean")
    # timer.start()
    if isinstance(data_path,Path):
        data = xr.open_dataset(data_path)
    else:
        data = open_mult(data_path)
    data_lon,data_lat = data["lon"].values, data["lat"].values
    vals = np.vstack([data[var].data for var in data.data_vars]).T
    resolution = 1/6
    t_resolution = np.array([timedelta(hours=3).seconds*1e9],dtype=np.int64)

    x_start=sign_add(x_boundary[0], resolution/2)
    x_end=sign_add(x_boundary[1], resolution/2)
    y_start=sign_add(y_boundary[0], resolution/2)
    y_end=sign_add(y_boundary[1], resolution/2)
    x_size = int((x_end-x_start)//resolution)
    y_size = int((y_end-y_start)//resolution)

    data_time = data["time"].values.astype(np.int64).copy()
    # t_start = np.array([data_time.min()],dtype=np.int64)

    t_start = (
        data_time
        .min()
        .astype("datetime64[ns]")
        .astype("datetime64[D]")
        .astype("datetime64[ns]")
        .astype(np.int64)
    )
    t_size = int(np.round((data_time.max() - t_start) / t_resolution))
    t_start = np.array([t_start],dtype=np.int64)
    
    block_mean = block_mean_loop_time(x_size,y_size,t_size,resolution,t_resolution,x_start,y_start,t_start,data_lon,data_lat,data_time,vals)

    # timer.stop()
    return block_mean

def make_interp_time(data_path: List[Path]) -> int:
    """Return interpolation time as integer value"""
    data=open_mult(data_path)
    times = data["time"].values
    mid_date = times.astype("datetime64[D]")[int(len(times)/2)].astype(str)
    mid_time = f"{mid_date}T12:00:00.000000000"
    return int(np.datetime64(mid_time)) # type: ignore

def setup_gridding(
        interp_lons: npt.NDArray[np.float64],
        interp_lats: npt.NDArray[np.float64],
        interp_time: int,
        land_mask: xr.Dataset,
        n_output_variables: int
    ) -> Tuple[npt.NDArray[np.float64],npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Setup input parameters for grid interpolation"""
    ocean_mask = ~np.isnan(land_mask.z.data)
    output = np.empty(list(ocean_mask.shape)+[n_output_variables], dtype=np.float64)
    output.fill(np.nan)
    times = np.ones(len(interp_lons.flatten()),dtype=np.int64)*interp_time
    interp_coords = np.vstack((interp_lons.flatten(),interp_lats.flatten(),times)).T
    ocean_mask_flat = ocean_mask.flatten()
    return output,interp_coords[ocean_mask_flat],ocean_mask

def grid_inter(
        interp_coords: npt.NDArray[np.float64],
        block_grid: npt.NDArray[np.float64]
    ) -> Tuple[int, npt.NDArray[np.float64] | None]:
    """Perform grid interpolation"""
    block_mean = block_grid[:,3:]
    coords = block_grid[:,:3]

    interpolator = RBFInterpolator(
        coords,
        block_mean,
        lat_column=1,
        lon_column=0,
        neighbors=500,
        kernel="linear",
        max_distance=500,
        min_points=5
    )
    try:
        return 0, interpolator(interp_coords)
    except ValueError:
        return 1, None

def store_attributes(
        masked_grid: xr.Dataset | npt.NDArray[np.float64],
        processed_file: List[Path],
        land_mask: xr.Dataset,
        interp_lons: npt.NDArray[np.float64],
        interp_lats: npt.NDArray[np.float64],
        interp_time: int
    ) -> xr.Dataset:
    """Transfer attributes from data netcdf to grid netcdf"""
    processed = open_mult(processed_file)
    if isinstance(masked_grid, np.ndarray):
        data_vars = list(processed.data_vars)
        layer_ids = range(masked_grid.shape[-1])
        masked_grid = xr.Dataset(
            data_vars={
                name:(['lats','lons'],masked_grid[:,:,i]) for name,i in zip(data_vars,layer_ids)
            },
            coords=dict(
                Longitude=(['lats','lons'],interp_lons),
                Latitude=(['lats','lons'],interp_lats),
                time=np.datetime64(interp_time, 'ns')
            )
        )
    masked_grid = masked_grid.assign_attrs(processed.attrs)
    res = land_mask.history.split('_')[2]
    file = processed_file[1].as_posix()
    root = file.split('/')[2].split('.')[0]
    grid_out_path = Path(f"Grids/{res}/3days/{root}_{res}.nc")
    masked_grid.to_netcdf(grid_out_path,mode="w")
    return masked_grid

def process_grid(land_mask: xr.Dataset, processed_file: List[Path], interp_lats: np.ndarray, interp_lons: np.ndarray) -> xr.Dataset:
    """Full grid processing pipeline"""
    block_grid = block_mean((-180,180),(-80,80),processed_file)
    interp_time = make_interp_time(processed_file)
    grid, interp_coords, ocean_mask = setup_gridding(interp_lons, interp_lats, interp_time, land_mask, block_grid.shape[1]-3)
    
    # timer = Timer("interpolation")
    # timer.start()
    status, output = grid_inter(interp_coords, block_grid)
    # timer.stop()
    if status != 0:
        return status, None
    grid[ocean_mask] = output
    final_grid=store_attributes(grid, processed_file, land_mask, interp_lons, interp_lats, interp_time)
    return status, final_grid

def open_mult(filepaths: List[Path]):
    """Open and concatenate multiple days of data as xarrays"""
    datasets=[xr.open_dataset(file, engine="netcdf4") for file in filepaths]
    return xr.concat(datasets,dim=list(datasets[0].dims)[0])

def file_to_date(file):
    """convert input file to date"""
    strs = file.name.split(".")[0].split("_")
    ints = list(map(int,strs))
    return date(year=ints[0],month=ints[1],day=ints[2])

class Timer:
    """Simple timer"""
    def __init__(self, function: str = ""):
        self.start_time = None
        self.end_time = None
        self.function = function
    
    def start(self):
        self.start_time = time.time()

    def stop(self):
        assert self.start_time != None, "Timer has not been started"
        self.end_time = time.time() - self.start_time
        if self.end_time < 60:
            print(f"Time elapsed ({self.function}): {self.end_time:.2f} s")
        if (self.end_time >= 60) & ((self.end_time)/60 < 60):
            print(f"Time elapsed ({self.function}): {(self.end_time)/60:.2f} min")
        if ((self.end_time)/60 >= 60):
            print(f"Time elapsed ({self.function}): {(self.end_time)/3600:.2f} h")
        

def main():
    timer = Timer("total")
    timer.start()
    # Paths
    PROCESSED = Path("Processed", "all")
    GRIDS = Path("Grids")
    GRIDS_01D = GRIDS / Path("01d")
    GRIDS_15M = GRIDS / Path("15m")
    GRIDS_10M = GRIDS / Path("10m")
    GRIDS_05M = GRIDS / Path("05m")
    GRIDS.mkdir(parents=True, exist_ok=True)
    GRIDS_01D.mkdir(parents=True, exist_ok=True)
    GRIDS_15M.mkdir(parents=True, exist_ok=True)
    GRIDS_10M.mkdir(parents=True, exist_ok=True)
    GRIDS_05M.mkdir(parents=True, exist_ok=True)
    files = PROCESSED.glob("2004_6_1[0-2].nc")
    
    dates = []
    for file in files:
        dt = file_to_date(file)
        dates.append(dt)
    dates.sort()
    
    files = []
    for i in range(1,len(dates)-1):
        d = []
        if dates[i]-timedelta(days=1) == dates[i-1]:
            d.append(dates[i-1])
        d.append(dates[i])
        if dates[i]+timedelta(days=1) == dates[i+1]:
            d.append(dates[i+1])

        fls = [PROCESSED / Path(f"{Date.year}_{Date.month}_{Date.day}.nc") for Date in d]
        files.append(fls)

    resolution_deg = 1 # 1, 1/4, 1/6 or 1/12
    land_mask_file = find_masking_attributes(resolution_deg)
    land_mask = xr.open_dataset(land_mask_file,engine="netcdf4").load()
    land_mask = subset_landmask(land_mask,(-180,180),(-80,80))
    
    interp_lons, interp_lats = make_grid(resolution_deg,resolution_deg,(-180,180),(-80,80))

    # Make commands
    commands: List[Tuple[xr.Dataset,List[Path],npt.NDArray[np.float64],npt.NDArray[np.float64]]] = []
    for file in files:
        commands.append((land_mask.copy(), file, interp_lats, interp_lons))
    
    for command in tqdm(commands):
        _ = process_grid(*command)
    # with multiprocessing.Pool() as pool:
    #     _ = pool.starmap(process_grid, commands)
    
    print("Complete")
    timer.stop()

if __name__ == "__main__":
    main()
