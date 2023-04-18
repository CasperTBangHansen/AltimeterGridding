import pandas as pd
from pathlib import Path
from typing import Iterable, List
import xarray as xr
from tqdm import tqdm

def load_ascii_file(file: Path, names: List[str]) -> xr.Dataset:
    """Loads an ascii file using pandas and converts it to a xarray dataset"""
    # Load data
    reader = pd.read_csv(file, sep=None, iterator=True, names=names, header=None, engine='python')
    data = reader.read().dropna()
    data['time'] = pd.to_datetime(data.time,format="%Y-%m-%d %H:%M:%S.%f")
    data.index = data['time'] # type: ignore
    return data.resample('1s').mean(numeric_only=True).dropna().to_xarray()

def convert_polar(output_folder: Path, folders: Iterable[Path]):
    """Loads the polar data in ascii format and translates them into netCDF files"""
    output_folder.mkdir(parents=True,exist_ok=True)

    for folder in folders:
        for measuring_type in (pbar := tqdm(folder.glob('*'))):
            for file in measuring_type.rglob('*.dat'):
                # Setup paths
                typename = measuring_type.name
                filename = file.name.split(".")[0].split("_")[-1]
                
                pbar.set_description(f"Processing [{folder.name}][{measuring_type.name}][{filename}]")
                out_path = output_folder / Path(f"{typename}_{filename}.nc")

                # Load data
                data = load_ascii_file(file, ["time","lon","lat","sla"])

                # Concat if file already exists
                if out_path.exists():
                    with xr.open_dataset(out_path) as current_data:
                        data = xr.concat([data, current_data], dim='time')

                # Export as xarray
                data.to_netcdf(out_path)
    
if __name__ == '__main__':
    # Paths
    arctic_path = Path("/g5/procdata/skr/to_ole/skr_c2_postproc_ascii")
    antarctic_path = Path("/g5/procdata/skr/to_ole/skr_postproc_ascii_sydpolen")
    convert_polar(Path("radsPolar"), [arctic_path, antarctic_path])
