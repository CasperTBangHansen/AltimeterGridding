
from sqlalchemy import create_engine, select, Row
from sqlalchemy.orm import sessionmaker, InstrumentedAttribute
from typing import Sequence, Optional, Tuple, Any, List, Dict
from os import environ
from . import tables
import logging
from datetime import date
import xarray as xr
from ..xarray_operations import encode
logging.getLogger(__name__)

class Database:
    def __init__(self, username: str, password: str, host: str, port: str | int, database_name: str, engine: str, database_type: str, create_tables: bool = True) -> None:
        self._url_object = f"{database_type}+{engine}://{username}:{password}@{host}:{port}/{database_name}?sslmode=require"
        self.engine = create_engine(self._url_object)
        self.session = sessionmaker(self.engine)
        if create_tables:
            tables.create_all_tables(self.engine)
        self.add_user(environ["DEFAULT_USERNAME"], environ["DEFAULT_PASSWORD"])

    def check_add(self, model: Any, expire_on_commit: bool = True, **kwargs) -> bool:
        """ Checks if the model alreay exists. If it does not it adds it"""
        with self.session(expire_on_commit=expire_on_commit) as session, session.begin():
            exists = session.scalars(select(type(model)).filter_by(**kwargs)).first()
            if exists:
                return False
            session.add(model)
        return True

    def add_product(self, product: tables.Product) -> bool:
        """ Add a product to the database"""        
        # Check id the product already exists
        name = product.name
        if self.check_add(product, name=name):
            logging.info(f"Adding {name} to the product table")
            return True
        logging.warning(f"{name} already existed in the product table")
        return False

    def add_user(self, username: str, password: str) -> bool:
        """Adds a user to the database"""
        username = username.lower()
        user = tables.User(username=username, password=tables.User.hash_password(password))
        if self.check_add(user, username=username):
            logging.info(f"Adding {username} to the user table")
            return True
        logging.warning(f"{username} already existed in the user table")
        return False

    def add_resolution(self, resolution: tables.Resolution) -> Tuple[bool, str]:
        """Adds a resolution to the database"""
        name = resolution.name
        prod_id = resolution.product_id
        with self.session() as session, session.begin():
            # Check product exists
            product_exists = select(tables.Product.id).filter_by(id=resolution.product_id)
            if session.scalars(product_exists).first() is None:
                logging.warning(f"No product matched product_id {prod_id}. {name} did not get added to the resolution table.")
                return False, f"{name} did not get added to the resolution table."
        # Add resolution
        if self.check_add(resolution, name=resolution.name):
            logging.info(f"Added {name} to the resolution table")
            return True, f"success"
        logging.warning(f"{name} already existed in the resolution table")
        return False, f"{name} already exists"

    def _construct_grid(self, grid: tables.Grid) -> bool:
        """Adds a grid to the database"""
        resolution_id = grid.resolution_id
        with self.session(expire_on_commit=False) as session, session.begin():
            # Check resolution exists
            resolution_exists = select(tables.Resolution.id).filter_by(id=grid.resolution_id)
            if session.scalars(resolution_exists).first() is None:
                logging.warning(f"No resolution matched resolution_id {resolution_id}. Did not add grid to the grid table.")
                return False
            grid.insert(session)
        # Add grid
        if self.check_add(grid, expire_on_commit=False):
            logging.info(f"Added grid to the grid table")
            return True
        logging.warning(f"Failed to add grid to the grid table")
        return False

    def _add_grid_id(self, dataset: xr.Dataset, day: date, resolution_id: int, srid: int = 4326) -> bool:
        """Adds a grid to the database"""
        # Make grid
        hexwkb = encode.dataset_to_hexwkb(dataset, srid=srid)
        strhexwkb = str(hexwkb)[2:-1] # remove b''
        grid = tables.Grid(
            raster=strhexwkb,
            date=day,
            references=str(dataset['z'].attrs.get('references')),
            ellipsoid=str(dataset['z'].attrs.get('ellipsoid')),
            ellipsoid_axis=dataset['z'].attrs.get('ellipsoid_axis'),
            ellipsoid_flattening=dataset['z'].attrs.get('ellipsoid_flattening'),
            mission_names=str(dataset['z'].attrs.get('mission_name')),
            mission_phase=str(dataset['z'].attrs.get('mission_phase')),
            rads=str(dataset['z'].attrs.get('title')),
            total_points=int(dataset['z'].attrs.get('total_points', 0)),
            n_points=str(list(dataset['z'].attrs.get('n_points', ''))),
            resolution_id=resolution_id
        )
        if(not self._construct_grid(grid)):
            return False
        return True

    def add_grid(self, dataset: xr.Dataset, day: date, resolution: int | tables.Resolution) -> bool:
        """Adds a grid to the database"""
        if isinstance(resolution, int):
            status = self._add_grid_id(dataset, day, resolution)
        else:
            status = self._add_grid_id(dataset, day, resolution.id)
        if status:
            logging.info("Added grid to the database")
        else:
            logging.warning("Failed to add the grid to the database")
        return status

    @staticmethod
    def _choose_id(
        id1: Sequence[int] | int | None,
        table1: InstrumentedAttribute[int],
        id2: Sequence[int] | int | None,
        table2: InstrumentedAttribute[int],
    ) -> Tuple[Sequence[int], InstrumentedAttribute[int]]:
        """Chooses the correct id/table pair based on if they are None or not"""
        if id1 is not None and id2 is not None:
            raise ValueError("both arguments can not be used at the same time")
        
        # Check input and get the correct table and id
        if id1 is not None:
            table = table1
            value: Sequence[int] | int = id1
        elif id2 is not None:
            table = table2
            value: Sequence[int] | int = id2
        else:
            raise ValueError("both ids can not be none as the same time")

        # Change from int to sequence of int
        if isinstance(value, int):
            value_seq: Sequence[int] = [value]
        else:
            value_seq: Sequence[int] = value
        return value_seq, table

    def delete_product(self, product: tables.Product) -> bool:
        """ Delete a product"""
        with self.session() as session, session.begin():
            # Get product id and Check if product already exists
            find_product = select(tables.Product.id).filter_by(name=product.name)
            product_id = session.scalars(find_product).all()
            if len(product_id) == 0:
                logging.warning(f"{product.name} did not exist")
                return False
            if not self.delete_resolutions(product_id=product_id):
                return False
            deleted_row = session.query(tables.Product).filter(tables.Product.id.in_(product_id)).delete()
            if deleted_row == 0:
                return False
            logging.info(f"Deleted {deleted_row} rows in the product table")
        return True

    def delete_resolutions(
        self,
        *,
        product_id: Optional[Sequence[int] | int] = None,
        resolution_id: Optional[Sequence[int] | int] = None
    ) -> bool:
        """ Delete resolutions and all the data mapped to that resolution"""
        # Get the corret table and id pairs
        value_seq, table_id = self._choose_id(resolution_id, tables.Resolution.id, product_id, tables.Resolution.product_id)

        with self.session() as session, session.begin():
            # Get resolutions for the product
            find_ids = select(tables.Resolution.id).filter(table_id.in_(value_seq))
            resolution_ids = session.scalars(find_ids).all()
            empty = len(resolution_ids) == 0
            if empty and product_id is not None:
                return True
            if empty:
                logging.warning(f"{value_seq} did not exist in the resolution table")
                return False
            if not self.delete_grid(resolution_id=resolution_ids):
                return False
            deleted_row = session.query(tables.Resolution).filter(tables.Resolution.id.in_(resolution_ids)).delete()
            if deleted_row == 0:
                return False
            logging.info(f"Deleted {deleted_row} rows in the resolution table")
        return True

    def delete_grid(
        self,
        *,
        resolution_id: Optional[Sequence[int] | int] = None,
        grid_id: Optional[Sequence[int] | int] = None
    ) -> bool:
        """ Delete grid and all the data mapped to that grid"""
        # Get the corret table and id pairs
        value_seq, table = self._choose_id(grid_id, tables.Grid.id, resolution_id, tables.Grid.resolution_id)

        with self.session() as session, session.begin():
            # Get grids for the resolutions
            find_ids = select(tables.Grid.id).filter(table.in_(value_seq))
            grid_id = session.scalars(find_ids).all()
            empty = len(grid_id) == 0
            if empty and resolution_id is not None:
                return True
            if empty:
                logging.warning(f"{value_seq} did not exist in the grid table")
                return False
            # Delete grid
            deleted_row = session.query(tables.Grid).filter(tables.Grid.id.in_(grid_id)).delete()
            if deleted_row == 0:
                return False
            logging.info(f"Deleted {deleted_row} rows in the grid table")
        return True

    def get_product_names(self) -> Sequence[str]:
        """ Get all products in the database"""
        logging.info("Getting all product names in database")
        with self.session(expire_on_commit=False) as session, session.begin():
            return session.scalars(select(tables.Product.name)).all()

    def get_product_by_name(self, product_name: str) -> tables.Product | None:
        """ Get products by name the database"""
        logging.info("Getting all product names in database")
        with self.session(expire_on_commit=False) as session, session.begin():
            return session.scalars(select(tables.Product).filter_by(name=product_name)).first()
    
    @staticmethod
    def row_to_dict(row: Row, keys: Sequence[str]) -> Dict[str, Any]:
        """Converts row object to a dict"""
        return {key: value for key, value in zip(keys, row)}

    def get_product_and_resolutions(self) -> List[Dict[str, str | float | int]]:
        """ Get all products which have a resolution and join the tables"""
        logging.info("Getting all product and resolutions in database")
        with self.session(expire_on_commit=False) as session, session.begin():
            # Define the select statement
            stmt = select(
                tables.Resolution.name.label('resolution_name'),
                tables.Resolution.x,
                tables.Resolution.y,
                tables.Resolution.time_days,
                tables.Product.name.label('product_name')
            ).join(tables.Product).where(tables.Resolution.product_id == tables.Product.id)
            
            # Get output keys            
            keys = [str(column).split('.')[-1] for column in stmt.columns]

            # Execute the query
            results = session.execute(stmt)
            
            # Convert results to list of dicts
            return [self.row_to_dict(r, keys) for r in results]

    def get_resolutions_by_name(self, resolution_name: str) -> tables.Resolution | None:
        """ Get all resolution by name in the database"""
        logging.info("Getting all resolution by names in database")
        with self.session(expire_on_commit=False) as session, session.begin():
            return session.scalars(select(tables.Resolution).filter_by(name=resolution_name)).first()

    def get_resolutions_by_product(
        self,
        product: Optional[tables.Product] = None,
        product_name: Optional[str] = None,
        product_id: Optional[str] = None,
    ) -> Sequence[tables.Resolution] | None:
        """ Get all resoltuons for a product in the database"""
        # Handle all combinations of input
        prod_id = None
        name = None
        if product is not None:
            if product.name is not None:
                name = product_name
            elif product.id is not None:
                prod_id = product_id
            else:
                raise ValueError("product was not None, but did not containg name or id")
        elif product_name is not None:
            name = product_name
        elif product_id is not None:
            prod_id = product_id
        else:
            raise ValueError("product, product_name and product_id where all None. One of them have to be provided")
        
        # Get resolution in table
        with self.session(expire_on_commit=False) as session, session.begin():
            if prod_id is None:
                prod_id = session.scalars(select(tables.Product.id).filter_by(name=name)).first()
            if prod_id is None:
                logging.warn(f"No product with the name '{name}' exists.")
                return None
            output = session.scalars(select(tables.Resolution).filter_by(product_id=prod_id)).all()
            if len(output) == 0:
                logging.warn(f"No resolution with product id {prod_id} exists.")
                return []
            logging.info(f"Found resolution with product_id = {prod_id} and resolution_id = {[i.id for i in output]}")
            return output
        return None

    def login(self, username: str, password: str) -> bool:
        """ Check if the user is in the database"""
        logging.info("Getting user from the database")
        user_db = None
        username = username.lower()
        with self.session(expire_on_commit=False) as session, session.begin():
            find_user = select(tables.User).filter_by(username=username)
            user_db = session.scalars(find_user).first()
        if user_db is None or not user_db.verify_password(username, password):
            logging.warning(f"Failed to login with {username}")
            return False
        logging.info(f"{username} successfully logged in")
        return True