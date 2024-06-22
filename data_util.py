import os
import numpy as np
import pandas as pd
import xarray as xr

__all__ = ["print_xarray", "save_with_progress"]


def print_xarray(ds, names=[]):
    
    if "time" in ds.dims:
        v = ds.isel(time=0)
    else:
        v = ds

    msg = f"name: {v.name}, shape: {v.shape}"

    if 'lat' in ds.dims:
        lat = ds.lat.values
        msg += f", lat: {lat[0]:.3f} ~ {lat[-1]:.3f}"
    if 'lon' in ds.dims:
        lon = ds.lon.values
        msg += f", lon: {lon[0]:.3f} ~ {lon[-1]:.3f}"   

    if "level" in v.dims:
        if len(names) > 0:
            v = v.sel(level=np.intersect1d(names, v.level))
        for lvl in v.level.values:
            x = v.sel(level=lvl).values
            msg += f"\nlevel: {lvl}, value: {x.min():.3f} ~ {x.max():.3f}"

    if "channel" in v.dims:
        if len(names) > 0:
            v = v.sel(channel=np.intersect1d(names, v.channel))        
        for ch in v.channel.values:
            x = v.sel(channel=ch).values
            msg += f"\nchannel: {ch}, value: {x.min():.3f} ~ {x.max():.3f}"

    print(msg + "\n")



def save_with_progress(ds, save_name, dtype=np.float32):
    from dask.diagnostics import ProgressBar

    if 'time' in ds.dims:
        ds = ds.assign_coords(time=ds.time.astype(np.datetime64))
    
    ds = ds.astype(dtype)

    if save_name.endswith(".nc"):
        obj = ds.to_netcdf(save_name, compute=False)
    elif save_name.endswith(".zarr"):
        obj = ds.to_zarr(save_name, compute=False)
    else:
        raise ValueError("save_type must be 'nc' or 'zarr'!")
    
    with ProgressBar():
        obj.compute()




