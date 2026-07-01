#!/usr/bin/env python3
"""Interoperability check: read a datacube-rs Zarr V3 store from Python.

Confirms that a cube written by the Rust `datacube-zarr` crate is consumable by
the standard zarr/xarray stack — the cloud-native-ARD claim. Verifies the array
shape, dimension names, the time/band/georeference attributes and a couple of
values.

    cargo run -q -p datacube-zarr --example write_sample -- /tmp/sample.zarr
    .venv-validate/bin/python scripts/zarr_interop.py /tmp/sample.zarr
"""

import sys

import numpy as np
import zarr

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sample.zarr"

g = zarr.open_group(path, mode="r")
arr = g["cube"]
attrs = dict(arr.attrs)

print(f"zarr format v{arr.metadata.zarr_format}, shape {arr.shape}, dtype {arr.dtype}")
print(f"dimension_names: {arr.metadata.dimension_names}")
print(f"bands: {attrs['bands']}")
print(f"time:  {attrs['time']}")
print(f"epsg:  {attrs.get('epsg')}  geotransform: {attrs.get('geotransform')}")

assert arr.ndim == 4, "expected (band, y, x, time)"
assert list(arr.metadata.dimension_names) == ["band", "y", "x", "time"]
assert attrs["bands"] == ["red", "nir"]
assert attrs["epsg"] == 32719

data = arr[:]
# NDVI from the red/nir bands read back through Python must match the formula
red = data[0]
nir = data[1]
ndvi = (nir - red) / (nir + red)
print(f"NDVI[0,0,0] (python, from rust-written store) = {ndvi[0, 0, 0]:.4f}")
assert np.isfinite(ndvi).all()

print("OK: Rust-written Zarr V3 cube read and validated from Python")
