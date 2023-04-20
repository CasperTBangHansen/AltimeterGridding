import subprocess
import multiprocessing
import os

from typing import List, Tuple, Iterable
from pathlib import Path
from tqdm import tqdm

import xarray as xr
import numpy as np
import numpy.typing as npt
from datetime import date, timedelta
from src import (
    RBFInterpolator,
    sign_add,
    _landmask_coord_bool,
    block_mean_loop_time,
    Timer,
    ExitCode,
    import_data
)

def find_masking_attributes(resolution_deg: float, base_path: Path) -> str:
    """Determine land masking from resolution"""
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

def block_mean_loop_time_(
    x_size: int,
    y_size: int,
    t_size: int,
    s_res: float,
    t_res: npt.NDArray[np.int64],
    x_start: float,
    y_start: float,
    t_start: npt.NDArray[np.int64],
    data_lon: npt.NDArray[np.float64],
    data_lat: npt.NDArray[np.float64],
    data_time: npt.NDArray[np.int64],
    vals) -> npt.NDArray[np.float64]:

    if len(data_lon) != len(vals):
        raise ValueError(f"Number of longitudes does not match number of values ({len(data_lon)} != {len(vals)})")
    if len(data_lat) != len(vals):
        raise ValueError(f"Number of latitudes does not match number of values ({len(data_lat)} != {len(vals)})")
    if len(data_time) != len(vals):
        raise ValueError(f"Number of times does not match number of values ({len(data_time)} != {len(vals)})")

    return block_mean_loop_time(x_size, y_size, t_size, s_res, t_res, x_start, y_start, t_start, data_lon, data_lat, data_time, vals)

def setup_spatial_grid_bounds(x_boundary: Tuple[float, float], y_boundary: Tuple[float, float], resolution: float) -> Tuple[float, float, int, int]:
    """Set the spatial boundaries for block mean grid"""
    x_start = sign_add(x_boundary[0], resolution/2)
    x_end = sign_add(x_boundary[1], resolution/2)
    y_start = sign_add(y_boundary[0], resolution/2)
    y_end = sign_add(y_boundary[1], resolution/2)
    x_size = int((x_end-x_start)//resolution)
    y_size = int((y_end-y_start)//resolution)
    return x_start, y_start, x_size, y_size

def setup_temporal_grid_bounds(data_time: npt.NDArray[np.int64], t_resolution: npt.NDArray[np.int64]):
    """Set the temporal boundaries for block mean grid"""
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
    return t_start, t_size

def remove_nans(data: xr.Dataset) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Remove nan values and return coordinates/values"""
    # Extract data input numpy arrays
    data_lon = data["lon"].values
    data_lat = data["lat"].values
    data_time = data["time"].values.astype(np.int64).copy()
    vals = np.vstack([data[var].data for var in data.data_vars]).T

    # Remove nan values
    remove_nan = ~np.isnan(vals).any(axis=1)
    vals = vals[remove_nan]
    data_lon = data_lon[remove_nan]
    data_lat = data_lat[remove_nan]
    data_time = data_time[remove_nan]
    return vals, data_lon, data_lat, data_time

def block_mean(
        x_boundary: Tuple[float, float],
        y_boundary: Tuple[float, float],
        data: xr.Dataset,
        temporal_resolution: int,
        spatial_resolution: float
    ) -> npt.NDArray[np.float64]:
    """mean grid in blocks of resolution size"""
    
    # Spatial and temporal resolution of block mean grid
    t_resolution = np.array(
        [timedelta(hours=temporal_resolution).seconds * 1e9],
        dtype=np.int64
    )

    vals, data_lon, data_lat, data_time = remove_nans(data)
    x_start, y_start, x_size, y_size = setup_spatial_grid_bounds(x_boundary, y_boundary, spatial_resolution)
    t_start, t_size = setup_temporal_grid_bounds(data_time, t_resolution)
    
    return block_mean_loop_time_(
        x_size,
        y_size,
        t_size,
        spatial_resolution,
        t_resolution,
        x_start,
        y_start,
        t_start,
        data_lon,
        data_lat,
        data_time,
        vals
    )

def make_interp_time(data_path: List[Path]) -> int:
    """Return interpolation time as integer value"""
    data = import_data(data_path)
    times = data["time"].values # type: ignore
    mid_date = times.astype("datetime64[D]")[int(len(times)/2)].astype(str)
    mid_time = f"{mid_date}T12:00:00.000000000"
    return int(np.datetime64(mid_time)) # type: ignore

def setup_gridding(
        interp_lons: npt.NDArray[np.float64],
        interp_lats: npt.NDArray[np.float64],
        interp_time: int,
        land_mask: xr.Dataset,
        n_output_variables: int
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Setup input parameters for grid interpolation"""
    # Ocean mask
    ocean_mask = ~np.isnan(land_mask.z.data)

    # Setup grid structure
    grid = np.empty(list(ocean_mask.shape) + [n_output_variables], dtype=np.float64)
    grid.fill(np.nan)

    # Get spatial and temporal components
    times = np.ones(len(interp_lons.flatten()), dtype=np.float64) * interp_time
    interp_coords = np.vstack((interp_lons.flatten(), interp_lats.flatten(), times)).T
    ocean_mask_flat = ocean_mask.flatten()

    return grid, interp_coords[ocean_mask_flat], ocean_mask

def grid_inter(
        interp_coords: npt.NDArray[np.float64],
        block_grid: npt.NDArray[np.float64],
        start_wrap: float = 160
    ) -> Tuple[ExitCode, npt.NDArray[np.float64] | None]:
    """Perform grid interpolation"""
   
    # Wrap coordinates
    wrapped_negative = block_grid[(block_grid[:,0] > abs(start_wrap))]
    wrapped_negative[:,0] -= 360
    wrapped_positive = block_grid[(block_grid[:,0] < -abs(start_wrap))]
    wrapped_positive[:,0] += 360
    block_grid = np.vstack([block_grid, wrapped_positive, wrapped_negative])

    # Split data
    block_mean = block_grid[:,3:] # [lon, lat, time]
    coords = block_grid[:,:3]

    try:
        interpolator = RBFInterpolator(
            coords,
            block_mean,
            lat_column=1,
            lon_column=0,
            neighbors=500,
            kernel="haversine",
            max_distance=500,
            min_points=5
        )
    except:
        return ExitCode.FAILURE, None
    return ExitCode.SUCCESS, interpolator(interp_coords)

def store_attributes(
        masked_grid: xr.Dataset | npt.NDArray[np.float64],
        processed_file: List[Path],
        interp_lons: npt.NDArray[np.float64],
        interp_lats: npt.NDArray[np.float64],
        interp_time: int
    ) -> xr.Dataset:
    """Transfer attributes from data netcdf to grid netcdf"""
    processed = import_data(processed_file)

    # Convert numpy array to xr.Dataset
    if isinstance(masked_grid, np.ndarray):
        data_vars = list(processed.data_vars)
        layer_ids = range(masked_grid.shape[-1])
        masked_grid = xr.Dataset(
            data_vars = {
                name:(['lats','lons'], masked_grid[:,:,i]) for name, i in zip(data_vars,layer_ids)
            },
            coords = dict(
                Longitude = (['lats','lons'], interp_lons),
                Latitude = (['lats','lons'], interp_lats),
                time = np.datetime64(interp_time, 'ns')
            )
        )
    
    # Transfer attributes from the original netcdf file 
    return masked_grid.assign_attrs(processed.attrs)

def process_grid(
        land_mask: xr.Dataset,
        processed_file: List[Path],
        interp_lats: npt.NDArray[np.float64],
        interp_lons: npt.NDArray[np.float64],
        grid_grouped_variables: List[List[str]],
        temporal_resolution: int,
        spatial_resolution: float,
        output_path_format: str
    ) -> ExitCode:
    """Full grid processing pipeline"""
    interp_time = None
    status = ExitCode.FAILURE

    # Load data
    all_data = import_data(processed_file)
    separated_grids = []
    for data in (all_data[g_var] for g_var in grid_grouped_variables): # type: ignore
        
        # Block mean the data
        block_grid = block_mean((-180,180), (-90,90), data, temporal_resolution, spatial_resolution)

        # Get time to interpolate to
        interp_time = make_interp_time(processed_file)

        # Setup gridding variables
        grid, interp_coords, ocean_mask = setup_gridding(interp_lons, interp_lats, interp_time, land_mask, block_grid.shape[1]-3)
        
        # Grid the data
        status, output = grid_inter(interp_coords, block_grid)
        if status != ExitCode.SUCCESS:
            return status
    
        # Perform oceanmasking of the output grid
        grid[ocean_mask] = output

        # Save each grid into a list to be concatenated
        separated_grids.append(grid)
    if interp_time is None:
        return ExitCode.FAILURE

    # Concatenate
    combined_grids = np.concatenate(separated_grids, axis=2)
    
    # Convert to netcdf for the correct attributes
    final_grid = store_attributes(combined_grids, processed_file, interp_lons, interp_lats, interp_time)
    
    # Export file
    date_str = processed_file[1].name.split('.')[0]
    grid_path = Path(output_path_format.format(date=date_str))
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    final_grid.to_netcdf(grid_path, mode="w", engine="netcdf4")

    return status

def file_to_date(file):
    """convert input file to date"""
    strs = file.name.split(".")[0].split("_")
    ints = list(map(int, strs))
    return date(year=ints[0], month=ints[1], day=ints[2])       

def group_valid_files(base_path: Path, files: Iterable[Path], n_days: int) -> List[Tuple[date, List[Path]]]:
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
            out_files.append((dates[i], fls))
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
    # CONST
    MULTIPROCESSING = True
    GRID_RESOLUTION = 1 # deg
    NUMBER_OF_DAYS = 3 # n days plus/minus
    BLOCKMEAN_SPATIAL_RESOLUTION = 1/6 # deg
    BLOCKMEAN_TEMPORAL_RESOLUTION = 3 # hours
    INTERPOLATION_GROUPS = [['sla'], ['sst', 'swh', 'wind_speed']]

    PIPELINE_VERSION = 4 # Pipeline version
    OUTPUT_GRID_PATH_FORMAT = "Grids/v{version}/{{date}}.nc".format(version=PIPELINE_VERSION) # Output format
    PROCESSED = Path(r"C:\Users\mathi\OneDrive\Dokumenter\DTU\Kandidat\Syntese\AltimeterGridding\Processed\Processed_v4") # Input folder
    DEFAULT_GLOB = "*.nc"

    timer = Timer("total")
    timer.Start()

    # Ocean mask
    land_mask_file = find_masking_attributes(GRID_RESOLUTION, Path("ocean_mask"))
    land_mask = xr.open_dataset(land_mask_file, engine="netcdf4").load()
    land_mask = subset_landmask(land_mask, (-180, 180), (-90, 90))
    
    # Construct interpolation coordinates
    interp_lons, interp_lats = make_grid(GRID_RESOLUTION, GRID_RESOLUTION, (-180, 180), (-90, 90))

    # Get correct glob
    # jobidx MUST BE A YEAR!
    files = adapt_file_list(processed=PROCESSED, default_glob=DEFAULT_GLOB, n_days=NUMBER_OF_DAYS)
        
    # Make commands
    commands = [
        (
            land_mask.copy(), file, interp_lats,
            interp_lons, INTERPOLATION_GROUPS,
            BLOCKMEAN_TEMPORAL_RESOLUTION, BLOCKMEAN_SPATIAL_RESOLUTION,
            OUTPUT_GRID_PATH_FORMAT
        )
        for file in files
    ]
    
    # Execute commands
    valid_commands: List[
        Tuple[
            xr.Dataset, List[Path], npt.NDArray[np.float64],
            npt.NDArray[np.float64], list[list[str]], int, float, str
        ]
    ] = []
    for command in tqdm(commands):
        date_str = command[1][1].name.split('.')[0]
        grid_path = Path(OUTPUT_GRID_PATH_FORMAT.format(date=date_str))
        if grid_path.exists():
            continue
        if MULTIPROCESSING:
            valid_commands.append(command)
        else:
            _ = process_grid(*command)
    if MULTIPROCESSING and valid_commands:
        if (command := valid_commands[0]):
            date_str = command[1][1].name.split('.')[0]
            print(f"Starting from {date_str}")
        with multiprocessing.Pool() as pool:
            _ = pool.starmap(process_grid, commands)
    
    print("Complete")
    timer.Stop()

if __name__ == "__main__":
    main()