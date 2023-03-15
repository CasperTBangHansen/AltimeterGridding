from pathlib import Path
from os import environ
from src import Database, tables
from datetime import date
import xarray as xr
from tqdm import tqdm

# Database
database = Database(
    username=environ["ALTIMETRY_USERNAME"],
    password=environ["ALTIMETRY_PASSWORD"],
    host=environ["ALTIMETRY_HOST"],
    port=environ["ALTIMETRY_DATABASE_PORT"],
    database_name=environ["ALTIMETRY_DATABASE"],
    engine=environ["ALTIMETRY_DATABASE_CONNECTION_ENGINE"],
    database_type=environ["ALTIMETRY_DATABASE_TYPE"],
    create_tables=environ["ALTIMETRY_CREATE_TABLES"] == 'true'
)

def setup_tables() -> None:
    product = tables.Product(name="RBF Interpolated (Latitude, Longitude)")
    if database.add_product(product):
        p_id = product.id
    else:
        prod = database.get_product_by_name(product.name)
        if prod is None:
            print("FAILED TO FIND PRODUCT")
            return
        p_id = prod.id
    resolution = tables.Resolution(product_id=p_id, x=1, y=1, time_days=1, name="Neighbors=100, kernel=linear")
    database.add_resolution(resolution)

def get_date_from_file_name(file_name):
    # split the file name into date components
    year, month, day, _ = file_name.split('_')
    # create a date object from the components
    date_obj = date(int(year), int(month), int(day))
    return date_obj


def upload_folder(folder: Path) -> None:
    resolution = database.get_resolutions_by_name("Neighbors=100, kernel=linear")
    if resolution is None:
        print("INVALID RESOLUTION")
        return
    files = list(folder.glob("*.nc"))
    files.sort()
    pbar = tqdm(files[::-1])
    for file in pbar:
        pbar.set_description(file.name)
        file_date = get_date_from_file_name(file.name)
        grid = xr.open_dataset(file)
        database.add_grid(dataset=grid, day=file_date, resolution=resolution)    


if __name__ == '__main__':
    # setup_tables()
    upload_folder(Path("Grids", "01d", "3days"))
