import subprocess
from typing import List, Tuple
from pathlib import Path
import multiprocessing
import os
import xarray as xr
import numpy as np
import numpy.typing as npt
import warnings
import time
from scipy.interpolate import RBFInterpolator
from datetime import date, timedelta, datetime
from grid_funcs import sign_add, _landmask_coord_bool, block_mean_loop_time
# from src.RBF_new._rbfinterp import RBFInterpolator

# @njit
# def block_mean_loop_time(
#     x_size,
#     y_size,
#     t_size,
#     s_res,
#     t_res,
#     start_pos_x,
#     start_pos_y,
#     start_pos_t,
#     data_lon,
#     data_lat,
#     data_time,
#     vals
# ):
#     lons = start_pos_x + np.arange(0,x_size+1)*s_res
#     lats = start_pos_y + np.arange(0,y_size+1)*s_res

#     times = start_pos_t + np.arange(0,t_size+1)*t_res
    
#     block_grid = np.zeros((len(vals),5))
#     count = 0
#     lookup = {}
#     for val,lon,lat,time in zip(vals,data_lon,data_lat,data_time):
#         idxlat = np.where(
#             (lat < lats+s_res/2) & (lat >= lats-s_res/2)
#         )[0]
#         idxlon = np.where(
#             (lon < lons+s_res/2) & (lon >= lons-s_res/2)
#         )[0]
#         idxtime = np.where(
#             (time < times+t_res/2) & (time >= times-t_res/2)
#         )[0]
        
#         for i in idxlon:
#             for j in idxlat:
#                 for t in idxtime:
#                     grididx = (i*(x_size+1)+j)*(t_size+1)+t
#                     if grididx in lookup:
#                         tmpidx = lookup.get(grididx,0)
#                         block_grid[tmpidx,0] += 1
#                         block_grid[tmpidx,1] += val
#                     else:
#                         lookup[grididx] = count
#                         block_grid[count,0] = 1
#                         block_grid[count,1] = val
#                         block_grid[count,2] = lons[i]
#                         block_grid[count,3] = lats[j]
#                         block_grid[count,4] = times[t]
#                         count += 1
#     block_grid = block_grid[:count]
#     block_grid[:,1] = block_grid[:,1] / block_grid[:,0]
#     return block_grid[:,1:]

def construct_sphinterpolate_command(
    file_path: Path,
    output_path: Path,
    variables: Tuple[str,...],
    resolution_deg: float,
    boundary: Tuple[float, float, float, float],
    verbose: bool = False
    ) -> List[str]:
    """Constructs a GMT sphinterpolate command (List of strings) based on input arguments"""
    boundary_shift = resolution_deg/2
    new_boundary = (
        boundary[0]+boundary_shift,
        boundary[1]-boundary_shift,
        boundary[2]+boundary_shift,
        boundary[3]-boundary_shift,
    )
    variables_str = "/".join(variables)
    boundary_str = "/".join(map(str, new_boundary))
    command: List[str] = [
        "gmt",
        "sphinterpolate",
        f"{file_path}?{variables_str}",
        f"-I{resolution_deg}",
        f"-G{output_path}",
        f"-R{boundary_str}"
    ]
    if verbose:
        command.append("-V3")
    return command

def find_masking_attributes(resolution_deg: float) -> str:
    """Determine land masking from resolution"""
    if resolution_deg == 1:
        land_mask_file = Path("data","land_NaN_01d.grd")
        command = f"gmt grdmath @earth_mask_01d_p 0 LE 0 NAN = {land_mask_file}"
    elif resolution_deg == 1/4:
        land_mask_file = Path("data","land_NaN_15m.grd")
        command = f"gmt grdmath @earth_mask_15m_p 0 LE 0 NAN = {land_mask_file}"
    elif resolution_deg == 1/6:
        land_mask_file = Path("data","land_NaN_10m.grd")
        command = f"gmt grdmath @earth_mask_10m_p 0 LE 0 NAN = {land_mask_file}"
    elif resolution_deg == 1/12:
        land_mask_file = Path("data","land_NaN_05m.grd")
        command = f"gmt grdmath @earth_mask_05m_p 0 LE 0 NAN = {land_mask_file}"
    else:
        raise ValueError("Invalid grid resolution. Valid resolutions are 1, 1/4, 1/6 or 1/12 degrees.")
    if not land_mask_file.is_file():
        subprocess.run(command, stdout=open(os.devnull, 'wb'))
    return land_mask_file.as_posix()

def create_grid(command: List[str]) -> subprocess.CompletedProcess[bytes]:
    """Process a commandline command"""
    output = subprocess.run(command, stdout=open(os.devnull, 'wb'))
    if output.returncode != 0:
        print(f"Failed to grid file: {command[2]}")
        return 1
    return 0

def make_grid(x_deg: float, y_deg: float, x_boundary: Tuple[float, float], y_boundary: Tuple[float, float]) -> Tuple[npt.NDArray[np.float64]]:
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

def block_mean(x_boundary: Tuple[float, float],y_boundary: Tuple[float, float],data_path: Path) -> npt.NDArray[np.float64]:
    """mean grid in blocks of resolution size"""
    # data = xr.open_dataset(data_path)
    timer = Timer("Block mean")
    timer.start()
    data = open_mult(data_path)
    data_lon,data_lat = data["lon"].values, data["lat"].values
    vals = data.sla.values
    resolution = 1/6
    t_resolution = int(timedelta(hours=12).seconds*1e9)

    x_start=sign_add(x_boundary[0], resolution/2)
    x_end=sign_add(x_boundary[1], resolution/2)
    y_start=sign_add(y_boundary[0], resolution/2)
    y_end=sign_add(y_boundary[1], resolution/2)
    x_size = int((x_end-x_start)//resolution)
    y_size = int((y_end-y_start)//resolution)

    data_time = data["time"].values.astype(np.int64)
    t_start = int(data_time.min())
    t_size = int(np.round((data_time.max() - t_start) / t_resolution))

    block_mean = block_mean_loop_time(x_size,y_size,t_size,resolution,t_resolution,x_start,y_start,t_start,data_lon,data_lat,data_time,vals)

    timer.stop()
    return block_mean

def make_interp_time(data_path: Path, hour_interval: int) -> np.ndarray:
    """Create array of times for interpolation"""
    data=open_mult(data_path)
    start = int(data["time"].values.min())
    end = int(data["time"].values.max())
    int_interval = int(timedelta(hours=hour_interval).seconds*1e9)
    return np.arange(start,end,int_interval)

def grid_inter(interp_lons: npt.NDArray[np.float64],interp_lats: npt.NDArray[np.float64],interp_times: npt.NDArray[np.int64], block_grid: npt.NDArray):#grid_lons: np.ndarray, grid_lats: np.ndarray):
    """Perform grid interpolation"""
    timer = Timer("interpolation")
    timer.start()
    block_mean = block_grid[:,0]
    coords = block_grid[:,1:]
    
    lut = RBFInterpolator(coords,block_mean.flatten(),neighbors=100,kernel="linear")#,distance='euclidean')#,callback=callback_print)

    grid = np.zeros((len(interp_lats.flatten()),len(interp_times)))
    for i,time in enumerate(interp_times):
        interp_time = np.ones(len(interp_lats.flatten()))*time
        interp_coords = np.stack((interp_lons.ravel(),interp_lats.ravel(),interp_time),-1)
        grid[:,i] = lut(interp_coords)
    
    if len(grid) != len(interp_lons.flatten()):
        print("Interpolation failed")
        return 1, None
    
    timer.stop()
    return 0, grid.reshape(interp_lats.shape[0],interp_lats.shape[1],len(interp_times)).shape

def mask_grid(grid: np.ndarray, land_mask: xr.Dataset):
    """Apply landmask to grid"""
    try:
        masked_grid = grid * land_mask.z
        if not masked_grid.shape == grid.shape:
            return 1,None
        return 0,masked_grid
    except ValueError:
        return 1, None

def store_attributes(masked_grid: xr.DataArray, processed_file: Path, land_mask: xr.Dataset):
    """Transfer attributes from data netcdf to grid netcdf"""
    # processed = xr.open_dataset(processed_file)
    processed = open_mult(processed_file)
    masked_grid = masked_grid.assign_attrs(processed.attrs)
    res = land_mask.history[24:27]
    file = processed_file[1].as_posix()
    root = file[14:-3]
    grid_out_path = Path(f"Grids/{res}/3days/{root}_{res}.nc")
    masked_grid.to_netcdf(grid_out_path,mode="w")
    return 0,masked_grid

# def insert_zero_rows(
#     block_mean_grid: npt.NDArray[np.float64],
#     lat: float
#     ) -> npt.NDArray[np.float64]:
#     """Insert row of zeroes at +/- the specified latitude in array"""
#     vals = np.zeros(block_mean_grid.shape[1])
#     block_mean_grid = np.append(np.insert(block_mean_grid,0,vals,0),vals.reshape(1,len(vals)),axis=0)
#     grid_lats = np.append(np.insert(grid_lats,0,-lat),lat)
#     return block_mean_grid, grid_lats

def process_grid(land_mask: xr.Dataset, processed_file: Path, interp_lats: np.ndarray, interp_lons: np.ndarray):
    """Create, mask and upload grid to container"""
    # status = create_grid(command)
    block_grid = block_mean((-180,180),(-80,80),processed_file)
    # block_grid = insert_zero_rows(block_grid,70)
    interp_times = make_interp_time(processed_file, hour_interval=12)
    status, grid=grid_inter(interp_lons,interp_lats,interp_times,block_grid)

    if status != 0:
        return 1
    status,masked_grid = mask_grid(grid, land_mask)
    if status != 0:
        return 1
    status,masked_grid=store_attributes(masked_grid, processed_file, land_mask)
    if status != 0:
        return 1
    
    return status,masked_grid

def open_mult(filepaths: List[Path]):
    """Open and concatenate multiple days of data as xarrays"""
    datasets=[xr.open_dataset(file, engine="netcdf4") for file in filepaths]
    return xr.concat(datasets,dim=list(datasets[0].dims)[0])

def file_to_date(file):
    """convert input file to date"""
    strs = file.name.split(".")[0].split("_")
    ints = list(map(int,strs))
    return date(year=ints[0],month=ints[1],day=ints[2])

def callback_print(iteration: int, total: int, message: bytes):
    """Callback function"""
    if (iteration == -1 and total == -1):
        print(message.decode())
    else:
        print(f"Progress {iteration:06d}/{total:06d} | {message.decode()}", end="\n" if iteration == total else '\r')

class Timer:
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
    files = PROCESSED.glob("2005_1*.nc")
    
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

    # gmt sphinterpolate txc11.nc?lon/lat/sla -I1 -Gout.grd -R-180/180/-90/90 -V3
    # Make commands
    commands: List[Tuple[str,str,str]] = []
    for file in files:
        commands.append((land_mask.copy(), file, interp_lats, interp_lons))
    
    for command in commands:
        process_grid(*command)
    # with multiprocessing.Pool() as pool:
    #     _ = pool.starmap(process_grid, commands)
    
    print("Complete")
    timer.stop()

if __name__ == "__main__":
    main()
