FORMAT_TYPES = {
    4: 'B', # PT_8BUI
    5: 'h', # PT_16BSI
    6: 'H', # PT_16BUI
    7: 'i', # PT_32BSI
    8: 'I', # PT_32BUI
    10: 'f', # PT_32BF
    11: 'd'  # PT_64BF
}

NP_STR_TO_WKT_RASTER_ID = {
    'int8': 4,
    'uint8': 4,
    'uint16': 6,
    'int16': 5,
    'uint32': 8,
    'int32': 7,
    'float32': 10,
    'float64': 11,
}

def numpy_dtype_to_wkt_raster_id(np_type: str) -> int:
    """Converts a numpy dtype (string) to wkt raster id type"""
    wkt_raster_id = NP_STR_TO_WKT_RASTER_ID.get(np_type)
    if wkt_raster_id is None:
        raise ValueError(f"Invalid dtype: {np_type} valid options are {list(NP_STR_TO_WKT_RASTER_ID)}")
    return wkt_raster_id

def pt2fmt(pt_id: int) -> str:
    """Returns binary data type specifier for given pixel type."""
    binary_type = FORMAT_TYPES.get(pt_id)
    if binary_type is None:
        raise ValueError(f"Invalid pixel type: {pt_id} valid options are {list(FORMAT_TYPES)}")
    return binary_type