from datetime import datetime
from sqlalchemy import ForeignKey, String, Engine, Double, Text, text
import sqlalchemy.orm as orm
import geoalchemy2 as geo
import bcrypt
from typing import List, Any, Dict, overload, Sequence
from os import environ

SALT: bytes = environ["SALT"].encode()
BaseClass: Any = orm.declarative_base() # type: ignore
Base: orm.DeclarativeMeta = BaseClass
class User(BaseClass):
    __tablename__ = "user"

    id: orm.Mapped[int] = orm.mapped_column(primary_key=True)
    username: orm.Mapped[str] = orm.mapped_column(Text, unique=True)
    password: orm.Mapped[str] = orm.mapped_column(Text)

    @staticmethod
    def hash_password(password: str) -> str:
        """ Hashes the password"""
        return bcrypt.hashpw(password.encode(), SALT).decode()

    def verify_password(self, username: str, password: str):
        """ Checks the password and username"""
        pwhash = bcrypt.checkpw(password.encode(), self.password.encode())
        return pwhash and self.username == username

class Product(BaseClass):
    __tablename__ = "product"

    id: orm.Mapped[int] = orm.mapped_column(primary_key=True)
    name: orm.Mapped[str] = orm.mapped_column(String(50), unique=True)
    # Resolution
    resolution: orm.Mapped[List["Resolution"]] = orm.relationship(back_populates="product")

class Resolution(BaseClass):
    __tablename__ = "resolution"
    
    # Fields
    id: orm.Mapped[int] = orm.mapped_column(primary_key=True)
    name: orm.Mapped[str] = orm.mapped_column(String(50), unique=True)
    x: orm.Mapped[float]
    y: orm.Mapped[float]
    time_days: orm.Mapped[int]
    # Product
    product_id: orm.Mapped[int] = orm.mapped_column(ForeignKey("product.id"))
    product: orm.Mapped["Product"] = orm.relationship(back_populates="resolution")
    # Grid
    grid: orm.Mapped[List["Grid"]] = orm.relationship(back_populates="resolution")

class Grid(BaseClass):
    __tablename__ = "grid"

    # Fields
    id: orm.Mapped[int] = orm.mapped_column(primary_key=True)
    raster = orm.mapped_column(geo.Raster(from_text = None, spatial_index=True))
    date: orm.Mapped[datetime]
    references: orm.Mapped[str] = orm.mapped_column(String(50))
    ellipsoid: orm.Mapped[str] = orm.mapped_column(String(20))
    ellipsoid_axis: orm.Mapped[float]
    ellipsoid_flattening: orm.Mapped[float] = orm.mapped_column(Double)
    mission_names: orm.Mapped[str] = orm.mapped_column(String(80))
    mission_phase: orm.Mapped[str] = orm.mapped_column(String(40))
    rads: orm.Mapped[str] = orm.mapped_column(String(30))
    total_points: orm.Mapped[int]
    n_points: orm.Mapped[str] = orm.mapped_column(String(40))

    # Grid
    resolution_id: orm.Mapped[int] = orm.mapped_column(ForeignKey("resolution.id"))
    resolution: orm.Mapped["Resolution"] = orm.relationship(back_populates="grid")

    def insert(self, session: orm.Session):
        statement = text(
            'INSERT INTO "grid" (date, raster, "references",'
            ' ellipsoid, ellipsoid_axis, ellipsoid_flattening, mission_names,'
            ' mission_phase, rads, resolution_id, total_points, n_points) VALUES'
            ' ((:date)::date, (:raster)::raster, :references, :ellipsoid,'
            ' :ellipsoid_axis, :ellipsoid_flattening, :mission_names,'
            ' :mission_phase, :rads, :resolution_id, :total_points, :n_points)'
        )
        session.execute(
            statement, # type: ignore
            {
                'date': self.date, 'raster': self.raster,
                'references': self.references, 'ellipsoid': self.ellipsoid,
                'ellipsoid_axis': self.ellipsoid_axis, 'ellipsoid_flattening': self.ellipsoid_flattening,
                'mission_names': self.mission_names, 'mission_phase': self.mission_phase,
                'rads': self.rads, 'resolution_id': self.resolution_id,
                'total_points': self.total_points, 'n_points': self.n_points
            }
        )

def create_all_tables(engine: Engine) -> None:
    """ Creates all tables if they dont exists"""
    Base.metadata.create_all(engine)

def delete_all_tables(engine: Engine) -> None:
    """ Deletes all the tables"""
    Base.metadata.drop_all(engine)

@overload
def get_fields_as_dict(table: orm.DeclarativeMeta) -> Dict[str, Any]:
    ...
@overload
def get_fields_as_dict(table: Sequence[orm.DeclarativeMeta]) -> List[Dict[str, Any]]:
    ...
def get_fields_as_dict(table: orm.DeclarativeMeta | Sequence[orm.DeclarativeMeta]) -> List[Dict[str, Any]] | Dict[str, Any]:
    """Converts a table to a dict containing the assigned values for each column"""
    if isinstance(table, Sequence):
        return [_get_fields_as_dict(x) for x in table]
    else:
        return _get_fields_as_dict(table)
def _get_fields_as_dict(table: orm.DeclarativeMeta) -> Dict[str, Any]:
    """Converts a table to a dict containing the assigned values for each column"""
    __table__ = getattr(table, '__table__', None)
    if __table__ is None:
        return {}
    return {column.name: getattr(table, column.name) for column in __table__.columns}
