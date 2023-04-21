import multiprocessing
from typing import List
from pathlib import Path
import xarray as xr
from src.interpolation import make_grid, process_grid, ConstructArguments
from src.fileHandler import locate_date_source, FileMapping
from src import Timer, config
from src.landmask import subset_landmask, find_masking_attributes


def grid(grid_arguments: ConstructArguments, file_mappings: List[FileMapping], output_format: str, multiprocess: bool = False, overwrite_grids: bool = False):
    for file_mapping in file_mappings:
        # Check if grid has already been processed
        grid_path = Path(output_format.format(date=file_mapping.computation_date_str))
        if grid_path.exists() and overwrite_grids:
            continue

        # Add filemapping to arguments
        grid_arguments.add_file_mapping(file_mapping)

        # Process later if multiprocessing is enabled otherwise process
        if not multiprocess:
            _ = process_grid(*grid_arguments.current_argument())

    # Execute commands using multiprocessing
    if multiprocess and grid_arguments:
        if (file := grid_arguments.first_file()) is not None:
            print(f"Starting from {file.computation_date}")
        with multiprocessing.Pool() as pool:
            _ = pool.starmap(process_grid, grid_arguments)
    

def main():
    # Load config files
    general, gridParameters, paths, interpolationParameters = config.parse_config(Path("config.ini"))

    # Make output format
    output_grid_path_format = paths.grid_path_format.format(version=general.pipeline_version)

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

    # Get all groups of files
    file_mappings = locate_date_source(processed=paths.raw_data_path, default_glob=paths.raw_data_glob, n_days=general.number_of_days)
    
    # Constructing arguments for gridding
    grid_arguments = ConstructArguments(
        land_mask, interp_lats, interp_lons,
        gridParameters, interpolationParameters, output_grid_path_format
    )

    # Construct grids
    timer = Timer("Gridding timer")
    grid(grid_arguments, file_mappings, output_grid_path_format, general.multiprocessing, general.overwrite_grids)
    timer.Stop()

if __name__ == "__main__":
    main()