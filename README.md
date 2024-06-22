## FuXi-ENS


This is the official repository for the FuXi-ENS paper.

FuXi-ENS: A machine learning model for medium-range ensemble weather forecasting

by Xiaohui Zhong, Lei Chen, Hao Li, Jun Liu, Xu Fan, Jie Feng, Kan Dai, Jie Wu, Bo Lu


## Installation
This document provides step-by-step instructions for setting up the necessary environment for our project.

### Step 1: Download the Required Files

The required files for this project, including the FuXi-ENS model and sample input data, are stored in a Google Drive folder. Please note that access to these resources is currently limited. If you need access to the Google Drive link, please contact Professor Li Hao at the following email address: lihao_lh@fudan.edu.cn.

The downloaded files shall be organized as the following hierarchy:

```plain
├── root
│   ├── data
│   │    ├── input.nc
│   │    ├── target.nc
|   |
│   ├── model
│   |    ├── fuxi_ens
│   |    ├── fuxi_ens.onnx
|   |   
│   ├── eval.py
│   ├── inference.py

```


### Step 2: Install xarray

Xarray is an open source project and Python package that makes working with labelled multi-dimensional arrays simple, efficient, and fun! It is particularly tailored to working with netCDF files.

You can install xarray along with dask, netCDF4, and bottleneck using the following command:

```bash
conda install -c conda-forge xarray dask netCDF4 bottleneck
```

### Step 3: Install ONNX Runtime

Use the following command to install ONNX Runtime:

```bash
pip install onnxruntime
```

If you want to use ONNX Runtime with GPU support, install the onnxruntime-gpu package instead:

```bash
pip install onnxruntime-gpu
```


## Usage
To use the model for inference, run the following command:

```python 
python inference.py \
    --model model/fuxi_ens.onnx \
    --input data/input.nc \
    --total_step 1 \
    --total_member 1 \
    --hour_interval 6 \
    --save_dir data/output;
```

To evaluate the model's performance, use the following command:

```python 
python eval.py \
    --target_dir data/ \
    --output_dir data/output \
    --hour_interval 6 \
    --test_chans z500 t850 \
    --test_steps 1 12 20 40 60;
```



## Input preparation 

The `input.nc` file contains preprocessed data from the origin ERA5 files. The file has a shape of **(2, 78, 121, 240)**, where the first dimension represents two time steps. The second dimension represents all variable and level combinations, named in the following exact order:



```python
['z50', 'z100', 'z150', 'z200', 'z250', 'z300', 'z400', 'z500',
'z600', 'z700', 'z850', 'z925', 'z1000', 't50', 't100', 't150',
't200', 't250', 't300', 't400', 't500', 't600', 't700', 't850',
't925', 't1000', 'u50', 'u100', 'u150', 'u200', 'u250', 'u300',
'u400', 'u500', 'u600', 'u700', 'u850', 'u925', 'u1000', 'v50',
'v100', 'v150', 'v200', 'v250', 'v300', 'v400', 'v500', 'v600',
'v700', 'v850', 'v925', 'v1000', 'q50', 'q100', 'q150', 'q200',
'q250', 'q300', 'q400', 'q500', 'q600', 'q700', 'q850', 'q925',
'q1000', 't2m', 'd2m', 'sst', 'u10m', 'v10m', 'u100m', 'v100m',
'msl', 'ssr', 'ssrd', 'fdir', 'ttr', 'tp']
```

The last 11 variables: **('t2m', 'd2m', 'sst', 'u10m', 'v10m', 'u100m', 'v100m',
'msl', 'ssr', 'ssrd', 'fdir', 'ttr', 'tp')** are surface variables, while the remaining variables represent atmosphere variables with numbers indicating pressure levels. The final five variables **('ssr', 'ssrd', 'fdir', 'ttr', 'tp')** are accumulated variables. These can be set to zero or left as they are, as the model will set them to zero during inference.
