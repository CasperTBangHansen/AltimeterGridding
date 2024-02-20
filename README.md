# Requirements

## Installation requirements
[RADS](https://github.com/remkos/rads) has to be installed on the system for downloading the satellite data. Install [GMT](https://www.generic-mapping-tools.org/) for getting the correct ocean masks. If [GMT](https://www.generic-mapping-tools.org/) is installed it would download the masks automatically. Make sure the python packages in the [requirements.txt](requirements.txt) file is installed. Additionally, the [environment.yml](environment.yml) file for creating a conda environment is provided, which is the recommended method of ensuring all required packages are installed as the requirements.txt file is not updated.


# Suggestion

## Pythran
Before running the scripts in the pipeline use ```pythran``` on  [grid_funcs.py](src/grid_funcs.py). This will make a ~100 times speedup on the block mean process. The resulting file should be placed in the same folder as [grid_funcs.py](src/grid_funcs.py).
```bash
pythran src/grid_funcs.py
```

# Pipeline
1. Generate the cycles for each satellite
```bash
./getAllRadsData.sh
```
Feel free to change the amount of cycles to download. Especially for the later satellites as they are still producing data.
2. Execute the following python script to process the data and put them into the correct formats
```python
python process_files.py
```
3. Execute the following python script for gridding the data.
```python
python grid.py
```
Change settings for grid interpolation in [config.ini](config.ini).

## Configuring the grid interpolation
The [config.ini](config.ini) file contains several methods of configurating the gridding. Some are more intuitive than others. Below is a brief explanation of each.

### General
**pipeline_version**: Assign a version to the resulting grid.

**multiprocessing**: Choose whether or not to use multiprocessing for the gridding. Provides a ~2x speed-up, however multiprocessing is severely more hardware-demanding especially memory-wise.

**number_of_days**: Assign the size of the moving temporal window over scattered data used for the interpolation of each grid day (+/- number of days).

**overwrite_grids**: If True, existing grids in output folder are overwritten.

### Paths
**grid_path_format**: Output folder

**raw_data_path**: Input folder.

**raw_data_glob**: Format for global search in input folder.

### Grid parameters
**grid_resolution**: Spatial resolution of output grid in degrees.

**block_grid_type**: Averaging method of block averaging before grid interpolation. Currently "mean" and "median" are implemented.

**blockmean_spatial_resolution**: Spatial resolution of block averaging.

**blockmean_temporal_resolution**: Temporal resolution of block averaging.

**interpolation_groups**: List of lists of which parameters to use for gridding. Implemented this way to enable different time/distance-weighting for different parameters.

**latitude_boundary**: Latitude boundary for which area is used for gridding.

**longitude_boundary**: Longitude boundary for which area is used for gridding.

### InterpolationParameters
**distance_to_time_scaling**: List of time/distance-weightings for each group of parameters.

**n_neighbors** Number of nearest neighbors (block-average-blocks) to use for each block in the grid interpolation.

**kernel**: Distance metric to use in interpolation. Options match those of the regular Scipy implementation of RBF interpolation with the addition of Haversine distance.

**max_distance_km**: The maximum distance to look for neighbors before dismissing a point in grid interpolation in km.

**min_points**: The minimum number of points within **max_distance_km** constituting a valid grid interpolation point. If less than this number of points are found, the grid point is discarded and replaced by NaN.
