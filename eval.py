import os
import shutil
import glob
import argparse
import numpy as np
import xarray as xr
import pandas as pd
from data_util import *

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--target_dir', type=str, required=True)
parser.add_argument('--test_chans', type=str, nargs='+', default=["z500", "t850", "msl", "t2m", "tp"])
parser.add_argument('--test_steps', type=int, nargs='+', default=[1, 12, 20, 40, 60])
parser.add_argument('--hour_interval', type=int, default=6)
parser.add_argument('--save_type', type=str, default="zarr", choices=["nc", "zarr"])
args = parser.parse_args()


def spatial_mean(x):
    weights = np.cos(np.deg2rad(np.abs(x.lat)))
    if "time" in x.dims:
        mean = x.weighted(weights).mean(("time", "lat", "lon"), skipna=True)
    else:
        mean = x.weighted(weights).mean(("lat", "lon"), skipna=True)    
    return mean


def ensemble_mean(out):
    if "normal" in out.dims:
        out = out.isel(normal=0, drop=True)
    if "member" in out.dims:
        out = out.mean("member", skipna=True)
    return out    


def compute_rmse(out, tgt):
    out = ensemble_mean(out)
    error = (out - tgt) ** 2 
    result = np.sqrt(spatial_mean(error))
    result.name = "rmse"
    return result


def merge_output(output_dir):
    save_type = args.save_type
    test_steps = args.test_steps
    test_chans = args.test_chans
    save_name = os.path.join(output_dir, f"output.{save_type}")

    if os.path.exists(save_name):
        os.remove(save_name)

    file_names = sorted(glob.glob(os.path.join(output_dir, f"**/*.{save_type}"), recursive=True))    
    assert len(file_names) > 0, f"No files found in {output_dir}"
    
    ds = xr.open_mfdataset(
        file_names, engine="zarr" if save_type == "zarr" else "netcdf4",
    ).chunk({'time': 1, 'step': 1})
    ds['channel'] = ds['channel'].str.lower()

    test_steps = np.intersect1d(ds.step, test_steps)
    test_chans = np.intersect1d(ds.channel, test_chans)
    ds = ds.sel(step=test_steps, channel=test_chans)


    save_with_progress(ds, save_name)


def align_step(out, tgt, step):
    hour_interval = args.hour_interval
    valid_times = out.time.data + pd.Timedelta(hours=step*hour_interval)
    _, ind1, ind2 = np.intersect1d(valid_times, tgt.time, return_indices=True)
    out = out.isel(time=ind1).sel(step=step, drop=True)
    tgt = tgt.isel(time=ind2)
    tgt = tgt.assign_coords(time=out.time)
    out = out.assign_coords(time=out.time)
    return out, tgt



if __name__ == "__main__":  
    output_dir = args.output_dir 
    target_dir = args.target_dir
    test_steps = args.test_steps
    test_chans = args.test_chans
    hour_interval = args.hour_interval

    output_file = os.path.join(output_dir, f"output.{args.save_type}")
    target_file = os.path.join(target_dir, f"target.{args.save_type}")

    merge_output(output_dir)

    assert os.path.exists(output_file), f"Output file {output_file} not found!"
    assert os.path.exists(target_file), f"Target file {target_file} not found!"

    output = xr.open_dataarray(output_file)
    target = xr.open_dataarray(target_file)

    # Convert channel names to lowercase
    output['channel'] = output['channel'].str.lower()
    target['channel'] = target['channel'].str.lower()

    test_chans = np.intersect1d(target.channel, test_chans)
    test_steps = np.intersect1d(output.step.data, test_steps)

    print(f"Test Channels: {test_chans}")
    print(f"Test Steps: {test_steps}")

    output = output.sel(channel=test_chans)
    target = target.sel(channel=test_chans)


    results = []
    for step in test_steps:
        out, tgt = align_step(output, target, step)
        result = compute_rmse(out, tgt)
        results.append(result)
    results = xr.concat(results, "step")
    
    results = results.assign_coords(step=test_steps)
    df = results.to_dataframe().unstack(0).T.round(2)    
    df.to_csv(os.path.join(output_dir, "rmse.csv"))
    print(f"{df}")


