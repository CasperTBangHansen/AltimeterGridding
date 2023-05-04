from typing import Tuple, List
import numpy as np
from numpy.linalg import LinAlgError
import numpy.typing as npt
import xarray as xr
from pathlib import Path
from . import ExitCode, make_interp_time, block_mean, RBFInterpolator
from ..fileHandler import import_data, FileMapping
from .. import config

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
        processed_file: FileMapping,
        interp_lats: npt.NDArray[np.float64],
        interp_lons: npt.NDArray[np.float64],
        gridParameters: config.GridParameters,
        interpolationParameters: config.InterpolationParameters,
        output_path_format: str
    ) -> ExitCode:
    """Full grid processing pipeline"""
    status = ExitCode.FAILURE

    # Load data
    all_data = import_data(processed_file.files)

    # Get time to interpolate to
    interp_time = make_interp_time(processed_file.computation_date)

    separated_grids: List[npt.NDArray[np.float64]] = []
    groups = (all_data[g_var] for g_var in gridParameters.interpolation_groups)
    for data, time_distance in zip(groups, interpolationParameters.distance_to_time_scaling):
        
        # Block mean the data
        block_grid = block_mean(
            (-180, 180),
            (-90, 90),
            data,
            gridParameters.blockmean_temporal_resolution,
            gridParameters.blockmean_spatial_resolution
        )

        # Setup gridding variables
        grid, interp_coords, ocean_mask = setup_gridding(interp_lons, interp_lats, interp_time, land_mask, block_grid.shape[1]-3)
        
        # Grid the data
        status, output = grid_inter(interp_coords, block_grid, interpolationParameters, time_distance)
        if status != ExitCode.SUCCESS:
            return status
    
        # Perform oceanmasking of the output grid
        grid[ocean_mask] = output

        # Save each grid into a list to be concatenated
        separated_grids.append(grid)

    # Concatenate
    combined_grids = np.concatenate(separated_grids, axis=2)
    
    # Convert to netcdf for the correct attributes
    final_grid = store_attributes(
        combined_grids,
        all_data,
        interp_lons,
        interp_lats,
        interp_time,
        [var for group in gridParameters.interpolation_groups for var in group]
    )
    
    # Export file
    grid_path = Path(output_path_format.format(date=processed_file.computation_date_str))
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    final_grid.to_netcdf(grid_path, mode="w", engine="netcdf4")

    return status

def grid_inter(
        interp_coords: npt.NDArray[np.float64],
        block_grid: npt.NDArray[np.float64],
        interpolationParameters: config.InterpolationParameters,
        time_distance: float,
    ) -> Tuple[ExitCode, npt.NDArray[np.float64] | None]:
    """Perform grid interpolation"""

    # Split data
    block_mean = block_grid[:,3:] # [lon, lat, time]
    coords = block_grid[:,:3]

    try:
        interpolator = RBFInterpolator(
            coords,
            block_mean,
            lat_column=1,
            lon_column=0,
            neighbors=interpolationParameters.n_neighbors,
            kernel=interpolationParameters.kernel,
            max_distance=interpolationParameters.max_distance_km,
            min_points=interpolationParameters.min_points,
            distance_to_time_scaling=time_distance
        )
        grid = interpolator(interp_coords)
    except LinAlgError as e:
        print(e)
        return ExitCode.FAILURE, None
    return ExitCode.SUCCESS, grid

def store_attributes(
        masked_grid: xr.Dataset | npt.NDArray[np.float64],
        raw_data: xr.Dataset,
        interp_lons: npt.NDArray[np.float64],
        interp_lats: npt.NDArray[np.float64],
        interp_time: int,
        data_vars: List[str]
    ) -> xr.Dataset:
    """Transfer attributes from data netcdf to grid netcdf"""

    # Convert numpy array to xr.Dataset
    if isinstance(masked_grid, np.ndarray):
        layer_ids = range(masked_grid.shape[-1])
        masked_grid = xr.Dataset(
            data_vars = {
                name:(['lats','lons'], masked_grid[:,:,i]) for name, i in zip(data_vars, layer_ids)
            },
            coords = dict(
                Longitude = (['lats','lons'], interp_lons),
                Latitude = (['lats','lons'], interp_lats),
                time = np.datetime64(interp_time, 'ns')
            )
        )
    
    # Transfer attributes from the original netcdf file 
    return masked_grid.assign_attrs(raw_data.attrs)