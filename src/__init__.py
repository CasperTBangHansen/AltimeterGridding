from ._rbfinterp import RBFInterpolator
from .shared_src import xarray_operations, Database, tables
from .grid_funcs import sign_add, _landmask_coord_bool, block_mean_loop_time
from .concatenateFiles import process_netcdfs, concatenate_date_netcdf_files
from .Timer import Timer
from .fileManipulation import import_data
from enum import Enum

class ExitCode(Enum):
    SUCCESS = 0
    FAILURE = 1