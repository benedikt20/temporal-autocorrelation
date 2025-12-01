# psd_proc.py
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import scipy
import spectrum
import os, yaml

# import functions for lowess processing
import lowess_funcs as lf

# suppress runtime warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================
# Config setup
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DATA_DIR = f"{config['data_dir']}/{config['data_subdirs']['psd_dir']}"
BANDWIDTH = config['bandwidth']  # bandwidht for MTM
K_TAPERS = config['k_tapers']    # number of tapers for MTM
CONF_RATIO = 1-10**config['conf_ratio_exp']  # confidence ratio for white noise detection
ERA5_DIR = f"{config['rawdata_dir']}/{config['rawdata_subdirs']['era5_dir']}"
YEARS = config['years']

# make dirs
os.makedirs(DATA_DIR, exist_ok=True)

# ==========================================
# process each year

for year in YEARS:
    # if there is a file with this year in the filename, skip
    if os.path.exists(f"{DATA_DIR}/spectral_slopes_{year}.nc"):
        print(f"Spectral slopes for year {year} exist..")
        continue

    print(f"Processing spectral slopes for year {year} ...")
    data = xr.open_dataset(f"{ERA5_DIR}/t2m_{year}.nc")

    def lowess_spectral_slopes(data_1d):
        lx, ly = lf.psd_lowess_proc(data_1d, conf_ratio=CONF_RATIO)
        breakpoint, slope1_br, slope2_br, rmse_br, slope1_daily, slope2_daily, rmse_daily, slope_lin, rmse_lin = lf.fit_breakpoint_glsar(lx, ly, plot=False)
        return breakpoint, slope1_br, slope2_br, rmse_br, slope1_daily, slope2_daily, rmse_daily, slope_lin, rmse_lin

    #subset = data['t2m'].isel(latitude=slice(20, 22), longitude=slice(120, 122))

    bps, br_slopes1, br_slopes2, br_rmses, daily_slopes1, daily_slopes2, daily_rmses, lin_slopes, lin_rmses = xr.apply_ufunc(
        lowess_spectral_slopes,    
        data['t2m'],
        input_core_dims=[['valid_time']],
        output_core_dims=[[], [], [], [], [], [], [], [], []], 
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float, float, float, float, float, float, float, float],
    )

    # ==========================================
    # setup xarray dataset for saving
    variables = {
        "breakpoint": (bps, "Breakpoint Frequency", "cycles/sample"),
        "slope1_br": (br_slopes1, "Spectral Slope 1 (BP Model)", ""),
        "slope2_br": (br_slopes2, "Spectral Slope 2 (BP Model)", ""),
        "rmse_br": (br_rmses, "RMSE (BP Model)", "log(Power)"),
        "slope1_daily": (daily_slopes1, "Spectral Slope 1 (Daily Breakpoint)", ""),
        "slope2_daily": (daily_slopes2, "Spectral Slope 2 (Daily Breakpoint)", ""),
        "rmse_daily": (daily_rmses, "RMSE (Daily Breakpoint)", "log(Power)"),
        "slope_lin": (lin_slopes, "Spectral Slope (Linear Model)", ""),
        "rmse_lin": (lin_rmses, "RMSE (Linear Model)", "log(Power)"),
    }

    data_vars = {}
    for var_name, (data_array, long_name, units) in variables.items():
        data_array.name = var_name
        data_array.attrs['long_name'] = long_name
        data_array.attrs['units'] = units
        data_vars[var_name] = data_array

    spectral_xr = xr.Dataset(data_vars)


    # ==========================================
    # save to data dir
    if os.path.exists(f"{DATA_DIR}/spectral_slopes_{year}.nc"):
        os.remove(f"{DATA_DIR}/spectral_slopes_{year}.nc")
    spectral_xr.to_netcdf(f"{DATA_DIR}/spectral_slopes_{year}.nc")