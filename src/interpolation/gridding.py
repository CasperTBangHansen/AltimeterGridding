from typing import Tuple, List
import numpy as np
import numpy.typing as npt
import xarray as xr
from pathlib import Path
from . import ExitCode, make_interp_time, block_mean, RBFInterpolator
from ..fileHandler import import_data

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

def process_grid(
        land_mask: xr.Dataset,
        processed_file: Tuple[str,List[Path]],
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
    all_data = import_data(processed_file[1])

    # Get time to interpolate to
    interp_time = make_interp_time(all_data)

    separated_grids = []
    for data in (all_data[g_var] for g_var in grid_grouped_variables): # type: ignore
        
        # Block mean the data
        block_grid = block_mean((-180,180), (-90,90), data, temporal_resolution, spatial_resolution)

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
    final_grid = store_attributes(combined_grids, all_data, interp_lons, interp_lats, interp_time)
    
    # Export file
    grid_path = Path(output_path_format.format(date=processed_file[0]))
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    final_grid.to_netcdf(grid_path, mode="w", engine="netcdf4")

    return status

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
        raw_data: xr.Dataset,
        interp_lons: npt.NDArray[np.float64],
        interp_lats: npt.NDArray[np.float64],
        interp_time: int
    ) -> xr.Dataset:
    """Transfer attributes from data netcdf to grid netcdf"""

    # Convert numpy array to xr.Dataset
    if isinstance(masked_grid, np.ndarray):
        data_vars = list(raw_data.data_vars)
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
    return masked_grid.assign_attrs(raw_data.attrs)