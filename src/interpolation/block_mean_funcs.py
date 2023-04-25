from . import sign_add, _landmask_coord_bool, block_mean_loop_time
from typing import Tuple, List
import numpy as np
import numpy.typing as npt
import xarray as xr
import datetime

def make_interp_time(interpolation_date: datetime.date) -> int:
    """Return interpolation time as integer value"""
    interpolation_time = datetime.datetime.combine(interpolation_date, datetime.time(hour=12))
    return np.datetime64(interpolation_time).astype(np.int64).item() * 1000

def make_grid(x_deg: float, y_deg: float, x_boundary: Tuple[float, float], y_boundary: Tuple[float, float]) -> List[npt.NDArray[np.float64]]:
    """Creates a grid of x, y"""
    x_start = sign_add(x_boundary[0], x_deg/2)
    x_end = sign_add(x_boundary[1], -x_deg/2)
    x = np.arange(x_start, x_end, x_deg)
    y_start = sign_add(y_boundary[0], y_deg/2)
    y_end = sign_add(y_boundary[1], -y_deg/2)
    y = np.arange(y_start, y_end, y_deg)
    return np.meshgrid(x, y)

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
        [datetime.timedelta(hours=temporal_resolution).seconds * 1e9],
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
