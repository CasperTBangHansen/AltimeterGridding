import subprocess
from typing import List, Tuple
from pathlib import Path
import multiprocessing
import os
import xarray as xr

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

def find_masking_attributes(resolution_deg: float):
    """Determine land masking from resolution"""
    if resolution_deg == 1:
        land_mask_file = Path("data","land_NaN_01d.grd")
    elif resolution_deg == 1/4:
        land_mask_file = Path("data","land_NaN_15m.grd")
    elif resolution_deg == 1/12:
        land_mask_file = Path("data","land_NaN_05m.grd")
    else:
        raise ValueError("Invalid grid resolution. Valid resolutions are 1, 1/4 or 1/12 degrees.")
    return land_mask_file.as_posix()

def create_grid(command: List[str]) -> subprocess.CompletedProcess[bytes]:
    """Process a commandline command"""
    output = subprocess.run(command, stdout=open(os.devnull, 'wb'))
    if output.returncode != 0:
        print(f"Failed to grid file: {command[2]}")
        return 1
    return 0

def mask_grid(grid_file: Path, land_mask: xr.Dataset):
    try:
        with xr.open_dataset(grid_file) as grid:
            if not grid:
                return 1,None
            land_mask=land_mask.reindex(lon=grid.lon,method='nearest',tolerance=1e-4)
            land_mask=land_mask.reindex(lat=grid.lat,method='nearest',tolerance=1e-4)
            masked_grid = grid.z * land_mask.z
            if not masked_grid.sizes == grid.sizes:
                return 1,None
        masked_grid.to_netcdf(grid_file,mode="w")
        if masked_grid:
            return 0,masked_grid
    except ValueError:
        return 1,None

def process_grid(command: List[str], grid_file: Path, land_mask: xr.Dataset):
    status = create_grid(command)
    if status != 0:
        return 1
    status,masked_grid = mask_grid(grid_file,land_mask)
    if status != 0:
        return 1
    return masked_grid
    

def main():
    # Paths
    PROCESSED = Path("Processed", "all")
    GRIDS = Path("Grids")
    GRIDS.mkdir(parents=True, exist_ok=True)
    files = PROCESSED.glob("*.nc")
    resolution_deg = 1/12
    land_mask_file = find_masking_attributes(resolution_deg)

    # with xr.open_dataset(land_mask_file) as land_mask:
    land_mask = xr.open_dataset(land_mask_file).load()
    # gmt sphinterpolate txc11.nc?lon/lat/sla -I1 -Gout.grd -R-180/180/-90/90 -V3
    # Make commands
    commands: List[Tuple[str,str,str]] = []
    for file in files:
        grid_path = GRIDS / Path(file.name)
        command = construct_sphinterpolate_command(
            file_path=file,
            output_path=grid_path,
            variables=("lon", "lat", "sla"),
            resolution_deg=resolution_deg,
            boundary=(-180, 180, -90, 90),
            verbose=False
        )
        commands.append((command, grid_path, land_mask.copy()))
    with multiprocessing.Pool() as pool:
        _ = pool.starmap(process_grid, commands)
    print("Complete")

if __name__ == "__main__":
    main()