from typing import Tuple
from .interpolation import _landmask_coord_bool
import subprocess
import xarray as xr
from pathlib import Path
from os import devnull

def subset_landmask(landmask: xr.Dataset, x_boundary: Tuple[float, float], y_boundary: Tuple[float, float]) -> xr.Dataset:
    """ Takes a subset of the landmask based on the x and y boundaries"""
    lat_min = _landmask_coord_bool(landmask.lat.values, y_boundary[0])
    lat_max = _landmask_coord_bool(landmask.lat.values, y_boundary[1])
    lon_min = _landmask_coord_bool(landmask.lon.values, x_boundary[0])
    lon_max = _landmask_coord_bool(landmask.lon.values, x_boundary[1])
    return landmask.isel(lat=slice(lat_min,lat_max), lon=slice(lon_min,lon_max))

def find_masking_attributes(resolution_deg: float, base_path: Path) -> str:
    """Determine land masking from resolution"""
    if not base_path.exists():
        base_path.mkdir()
    if resolution_deg == 1:
        land_mask_file = Path("land_NaN_01d.grd")
        mask_name = "earth_mask_01d_p"
    elif resolution_deg == 1/2:
        land_mask_file = Path("land_NaN_30m.grd")
        mask_name = "earth_mask_30m_p"
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
        raise ValueError("Invalid grid resolution. Valid resolutions are 1, 1/2, 1/4, 1/6 or 1/12 degrees.")
    
    land_mask_file = base_path / land_mask_file
    if not land_mask_file.is_file():
        command = f"gmt grdmath @{mask_name} 0 LE 0 NAN = {land_mask_file}"
        subprocess.run(command, stdout=open(devnull, 'wb'))
    return land_mask_file.as_posix()
