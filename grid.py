import subprocess
from typing import List, Tuple, Iterable
from pathlib import Path
import multiprocessing
import os

def construct_sphinterpolate_command(
    file_path: Path,
    output_path: Path,
    variables: Tuple[str,...],
    resolution_deg: float,
    boundary: Tuple[float, float, float, float],
    verbose: bool = False
    ) -> List[str]:
    """Constructs a GMT sphinterpolate command (List of strings) based on input arguments"""
    variables_str = "/".join(variables)
    boundary_str = "/".join(map(str, boundary))
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

def process_commands(command: List[str]) -> subprocess.CompletedProcess[bytes]:
    """Process a commandline command"""
    output = subprocess.run(command, stdout=open(os.devnull, 'wb'))
    if output.returncode != 0:
        print(f"Failed to grid file: {command[2]}")
    return output

def main():
    # Paths
    PROCESSED = Path("Processed", "all")
    GRIDS = Path("Grids")
    GRIDS.mkdir(parents=True, exist_ok=True)
    files = PROCESSED.glob("*.nc")

    # gmt sphinterpolate txc11.nc?lon/lat/sla -I1 -Gout.grd -R-180/180/-90/90 -V3
    # Make commands
    commands: List[str] = []
    for file in files:
        commands.append(
            construct_sphinterpolate_command(
                file_path=file,
                output_path=GRIDS / Path(file.name),
                variables=("lon", "lat", "sla"),
                resolution_deg=1,
                boundary=(-180, 180, -90, 90),
                verbose=False
            )
        )
    with multiprocessing.Pool() as pool:
        return_codes = pool.map(process_commands, commands)
    print("Complete")

if __name__ == "__main__":
    main()