import argparse
import os
import glob
import numpy as np
import xarray as xr
import pandas as pd
import onnxruntime as ort
from time import perf_counter
from copy import deepcopy
from data_util import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, 
                    help="Path to the ONNX file for the FuXi-S2S model.")
parser.add_argument('--input', type=str, required=True, 
                    help="Path to the input NetCDF data file.")
parser.add_argument('--save_dir', type=str, required=True, 
                    help="Directory where the prediction output will be saved.")
parser.add_argument('--save_type', type=str, default="nc", choices=["nc", "zarr"])
parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"])
parser.add_argument('--total_step', type=int, default=1)
parser.add_argument('--total_member', type=int, default=1)
parser.add_argument('--hour_interval', type=int, default=6)
args = parser.parse_args()


def save_pred(output, input, step, member=0):
    save_type = args.save_type
    save_dir = args.save_dir

    tmp_dir = os.path.join(save_dir, f"member_{member:03d}")
    os.makedirs(tmp_dir, exist_ok=True)

    init_time = pd.to_datetime(input.time.data[-1])
    out = xr.DataArray(
        name="data",
        data=output[..., np.newaxis],
        dims=['time', 'step', 'channel', 'lat', 'lon', 'member'],
        coords=dict(
            time=[init_time],
            step=[step],
            channel=input.channel,
            lat=input.lat,
            lon=input.lon,
            member=[member],
        )
    ).astype(np.float32)
    print_dataarray(out)
    save_name = os.path.join(tmp_dir, f'{step:03d}.{save_type}')
    save_with_progress(out, save_name)

    
def load_model(model_name, device):
    ort.set_default_logger_severity(3)
    options = ort.SessionOptions()
    # Enable memory pattern optimization for speed
    options.enable_mem_pattern = True
    # Enable CPU memory arena for speed
    options.enable_cpu_mem_arena = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    if device == "cuda":
        providers = [('CUDAExecutionProvider', {'arena_extend_strategy':'kSameAsRequested'})]
    elif device == "cpu":
        providers=['CPUExecutionProvider']
        options.intra_op_num_threads = os.cpu_count() or 1
    else:
        raise ValueError("device must be cpu or cuda!")

    session = ort.InferenceSession(
        model_name,  
        sess_options=options, 
        providers=providers
    )
    return session


def run_inference(model,  input):
    total_step = args.total_step
    total_member = args.total_member
    hour_interval = args.hour_interval

    hist_time = pd.to_datetime(input.time.values[-2])
    init_time = pd.to_datetime(input.time.values[-1])
    assert init_time - hist_time == pd.Timedelta(hours=hour_interval)

    lat = input.lat.values 
    batch = input.values[None]
    assert lat[0] == 90 and lat[-1] == -90
    
    print(f"\nInference process at {init_time} ...")

    for member in range(total_member):
        first_time = perf_counter()
        print(f'\nInference for member {member:03d} ...')
        new_input = deepcopy(batch)

        for t in range(total_step):
            valid_time = init_time + pd.Timedelta(hours=t * hour_interval)
            inputs = {'input': new_input}        

            if "step" in input_names:
                inputs['step'] = np.array([t], dtype=np.float32)

            if "hour" in input_names:
                hour = [valid_time.hour/24]
                # print(f"hour: {hour}")
                inputs['hour'] = np.array(hour, dtype=np.float32)     

            if "doy" in input_names:
                doy = min(365, valid_time.day_of_year)/365 
                # print(f"doy: {doy}")
                inputs['doy'] = np.array([doy], dtype=np.float32)

            start_time = perf_counter()
            new_input, = model.run(None, inputs)
            output = deepcopy(new_input[:, -1:])
            save_pred(output, input, t+1,  member)
            elapsed_time = perf_counter() - start_time
            
            print(f"member: {member:03d}, step {t+1:03d}, time: {elapsed_time:.3f} secs")
            
            if t > total_step:
                break

        total_time = perf_counter() - first_time
        print(f'Inference for member {member:03d} done, take {total_time:.3f} secs')

    print(f"\nInference process at {init_time} done")


if __name__ == "__main__":
    assert os.path.exists(args.input), f"Input file {args.input} not found!"
    assert os.path.exists(args.model), f"Model file {args.model} not found!"

    input = xr.open_dataarray(args.input)
    print_dataarray(input)        
    
    print(f'Load FuXi ...')       
    start = perf_counter()
    model = load_model(args.model, args.device)
    input_names = [input.name for input in model.get_inputs()]
    print(f"{input_names=}")
    print(f'Load FuXi take {perf_counter() - start:.2f} secs')
    
    run_inference(model, input)




