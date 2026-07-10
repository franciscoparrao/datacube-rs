#!/usr/bin/env python3
"""Interoperability check: read a datacube-rs Zarr V3 store from Python.

Confirms that a cube written by the Rust `datacube-zarr` crate is consumable by
the standard zarr/xarray/pyproj stack — the cloud-native-ARD claim. Verifies
the array shape, dimension names, the time/band/georeference attributes, the
GeoZarr-CF companions (`/y`, `/x`, `/time` coordinate arrays and the
`/spatial_ref` grid-mapping variable), and a couple of values.

    cargo run -q -p datacube-zarr --example write_sample -- /tmp/sample.zarr
    .venv-validate/bin/python scripts/zarr_interop.py /tmp/sample.zarr
"""

import sys

import numpy as np
import xarray as xr
import zarr
from pyproj import CRS

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sample.zarr"

g = zarr.open_group(path, mode="r")
arr = g["cube"]
attrs = dict(arr.attrs)

print(f"zarr format v{arr.metadata.zarr_format}, shape {arr.shape}, dtype {arr.dtype}")
print(f"dimension_names: {arr.metadata.dimension_names}")
print(f"bands: {attrs['bands']}")
print(f"time:  {attrs['time']}")
print(f"epsg:  {attrs.get('epsg')}  geotransform: {attrs.get('geotransform')}")
print(f"band_long_names: {attrs.get('band_long_names')}")
print(f"grid_mapping: {attrs.get('grid_mapping')}")

assert arr.ndim == 4, "expected (band, y, x, time)"
assert list(arr.metadata.dimension_names) == ["band", "y", "x", "time"]
assert attrs["bands"] == ["red", "nir"]
assert attrs["epsg"] == 32719
assert attrs["band_long_names"] == ["red", "nir"]  # unknown names fall back to themselves
assert attrs["grid_mapping"] == "spatial_ref"

data = arr[:]
# NDVI from the red/nir bands read back through Python must match the formula
red = data[0]
nir = data[1]
ndvi = (nir - red) / (nir + red)
print(f"NDVI[0,0,0] (python, from rust-written store) = {ndvi[0, 0, 0]:.4f}")
assert np.isfinite(ndvi).all()

# GeoZarr-CF companions: separate coordinate variables + grid-mapping variable.
y, x, time = g["y"], g["x"], g["time"]
print(f"y[:3]: {y[:3]}  x[:3]: {x[:3]}  time: {time[:]}")
assert dict(y.attrs)["standard_name"] == "projection_y_coordinate"
assert dict(x.attrs)["standard_name"] == "projection_x_coordinate"
assert dict(time.attrs)["standard_name"] == "time"
assert y.shape == (arr.shape[1],)
assert x.shape == (arr.shape[2],)
# pixel-center coordinates from the geotransform: origin (300000, 6200000), 10m pixels
assert np.isclose(x[0], 300000.0 + 5.0)
assert np.isclose(y[0], 6200000.0 - 5.0)

spatial_ref_attrs = dict(g["spatial_ref"].attrs)
print(f"spatial_ref.grid_mapping_name: {spatial_ref_attrs.get('grid_mapping_name')}")
crs = CRS.from_wkt(spatial_ref_attrs["crs_wkt"])
print(f"pyproj CRS parsed from crs_wkt: {crs.to_epsg()} ({crs.name})")
assert crs.to_epsg() == 32719
assert spatial_ref_attrs["grid_mapping_name"] == "transverse_mercator"
assert spatial_ref_attrs["GeoTransform"] == "300000 10 0 6200000 0 -10"


# The real payoff of separate coordinate variables + dimension_names: xarray
# opens the store and auto-recognizes y/x/time as coordinates with no extra
# decoding hints, the way it would for a store any other GeoZarr writer made.
ds = xr.open_zarr(path, consolidated=False)
print(f"xarray coords: {sorted(ds.coords)}")
assert {"y", "x", "time"}.issubset(ds.coords)
assert ds["cube"].attrs["grid_mapping"] == "spatial_ref"
assert np.allclose(ds.coords["y"].values, y[:])

print("OK: Rust-written Zarr V3 cube (with GeoZarr-CF coordinates + grid_mapping) read and validated from Python")
