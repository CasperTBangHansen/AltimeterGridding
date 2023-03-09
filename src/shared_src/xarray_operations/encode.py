import xarray as xr
import binascii
import struct
from .sizes import transform
from . import dtypes

def dataset_to_hexwkb(dataset: xr.Dataset, srid: int) -> bytes:
    """Encodes a dataset into HEX-encoded WKB for WKT rasters."""
    # Header in bytes 
    hexwkb = raster_header_to_hexwkb(dataset, srid)

    # Add data in bytes to the header
    for key in dataset.data_vars:
        band = dataset[key]
        hexwkb += band_header_to_hexwkb(band)
        hexwkb += wkblify_band(band)
    return hexwkb

def wkblify_band(band: xr.DataArray) -> bytes:
    """Writes band of given xr.DataArray into HEX-encoded WKB for WKT Raster output."""
    pixels = band.values
    if pixels is None:
        raise ValueError("Invalid array was read from band")
    return binascii.hexlify(pixels) # type: ignore

def raster_header_to_hexwkb(dataset: xr.Dataset, srid: int) -> bytes:
    """Encodes the header of the dataset to a hexwkb format"""
    xsize, ysize = dataset.sizes.values()
    gt = transform(dataset)
    rt_ip = (gt[0], gt[3])
    rt_skew = (gt[2], gt[4])
    rt_scale = (gt[1], gt[5])
    raster_count = len(dataset.data_vars)

    # Encodes:
    # Endiannes, Version, Band and georeferences

    # Endiannes
    hexwkb: bytes = wkblify('B', 1) # 1 = Little-endian
    # Version
    hexwkb += wkblify('H', 0) # 0 = WKTRaster version

    # Bands
    hexwkb += wkblify('H', raster_count)
    # Georeference
    hexwkb += wkblify('d', rt_scale[0])
    hexwkb += wkblify('d', rt_scale[1])
    hexwkb += wkblify('d', rt_ip[0])
    hexwkb += wkblify('d', rt_ip[1])
    hexwkb += wkblify('d', rt_skew[0])
    hexwkb += wkblify('d', rt_skew[1])
    hexwkb += wkblify('i', srid)

    # Number of columns and rows
    hexwkb += wkblify('H', xsize)
    hexwkb += wkblify('H', ysize)
    return hexwkb

def band_header_to_hexwkb(data_array: xr.DataArray) -> bytes:
    """Encodes the band header to a hexwkb format"""
    # Encode dtype of array
    pixeltype = dtypes.numpy_dtype_to_wkt_raster_id(str(data_array.dtype))

    # Encodes pixel and nodata value
    # It has been assumed there is data it every point
    hexwkb = wkblify('B', 64 + pixeltype)
    hexwkb += wkblify(dtypes.pt2fmt(pixeltype), 0)
    return hexwkb

def wkblify(fmt: str, data: int | float) -> bytes:
    """Writes raw binary data into HEX-encoded string using binascii module."""
    # Binary to HEX
    fmt_little = '<' + fmt
    hexstr = binascii.hexlify(struct.pack(fmt_little, data)).upper()
    return hexstr