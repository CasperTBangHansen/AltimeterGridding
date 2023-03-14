from struct import unpack
import numpy as np
import xarray as xr
from datetime import timedelta


__all__ = [
    'read_wkb_raster',
    'raster_to_xarray'
]

def raster_to_xarray(raster):
    decoded_data = read_wkb_raster(raster.raster.data)
    data_vars = ['sla', 'sst', 'swh', 'wind_speed']
    lats = np.arange(-80+decoded_data['scaleY']/2,80+decoded_data['scaleY']/2,decoded_data['scaleY'])
    lons = np.arange(-180+decoded_data['scaleX']/2,180+decoded_data['scaleX']/2,decoded_data['scaleX'])

    return xr.Dataset(
        data_vars={
            name:(['lats','lons'], band['ndarray']) for name, band in zip(data_vars, decoded_data['bands'])
        },
        coords=dict(
            Longitude=(['lons'], lons),
            Latitude=(['lats'], lats),
            time=np.datetime64(raster.date + timedelta(hours=12), 'ns')
        ),
        attrs={key: value for key, value in raster.__dict__.items() if key not in ['raster', 'date', 'id', 'resolution_id', '_sa_instance_state']}
    )

def read_wkb_raster(wkb):
    """Read a WKB raster to a Numpy array."""
    wkb = bytes(bytearray.fromhex(wkb))
    ret = {}

    # Determine the endiannes of the raster
    (endian,) = unpack('<b', wkb[:1])

    if endian == 0:
        endian = '>'
    elif endian == 1:
        endian = '<'

    # Read the raster header data.
    (version, bands, scaleX, scaleY, ipX, ipY, skewX, skewY,
     srid, width, height) = unpack(endian + 'HHddddddIHH', wkb[1:61])

    ret['version'] = version
    ret['scaleX'] = scaleX
    ret['scaleY'] = scaleY
    ret['ipX'] = ipX
    ret['ipY'] = ipY
    ret['skewX'] = skewX
    ret['skewY'] = skewY
    ret['srid'] = srid
    ret['width'] = width
    ret['height'] = height
    ret['bands'] = []

    position = 61
    for _ in range(bands):
        band = {}
        # Requires reading a single byte, and splitting the bits into the
        # header attributes
        (bits,) = unpack(endian + 'b', wkb[position:position+1])
        position += 1

        band['isOffline'] = bool(bits & 128)  # first bit
        band['hasNodataValue'] = bool(bits & 64)  # second bit
        band['isNodataValue'] = bool(bits & 32)  # third bit

        pixtype = bits & 15  # bits 5-8

        # Based on the pixel type, determine the struct format, byte size and
        # numpy dtype
        fmts = ['?', 'B', 'B', 'b', 'B', 'h',
                'H', 'i', 'I', 'f', 'd', 'd']
        dtypes = ['b1', 'u1', 'u1', 'i1', 'u1', 'i2',
                  'u2', 'i4', 'u4', 'f4', 'f8', 'f8']
        sizes = [1, 1, 1, 1, 1, 2, 2, 4, 4, 4, 8, 8]

        dtype = dtypes[pixtype]
        size = sizes[pixtype]
        fmt = fmts[pixtype]

        # Read the nodata value
        (nodata,) = unpack(endian + fmt, wkb[position:position+size])
        position += size

        band['nodata'] = nodata

        if band['isOffline']:

            # Read the out-db metadata

            # offline bands are 0-based, make 1-based for user consumption
            (band_num,) = unpack(endian + 'B', wkb[position:position+1])
            position += 1
            band['bandNumber'] = band_num + 1

            data = b''
            while True:
                byte = wkb[position:position+1]
                position += 1
                if byte == b'\x00':
                    break

                data += byte

            band['path'] = data.decode()

        else:

            # Read the pixel values: width * height * size
            band['ndarray'] = np.ndarray(
                (width, height),
                buffer=wkb[position:position+width * height * size],
                dtype=np.dtype(dtype)
            )
            position += width * height * size

        ret['bands'].append(band)

    return ret