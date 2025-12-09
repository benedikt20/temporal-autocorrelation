# landcover_proc.py
import pandas as pd
import yaml

import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin, Affine
import numpy as np
import glob
import os, re


# ==========================================================
# config setup
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

LANDCOVER_LEGEND_DIR = config['landcover_legend_dir']
TIF_DIR = f"{config['rawdata_dir']}/{config['rawdata_subdirs']['landcover_dir']}"
PSD_DIR = f"{config['data_dir']}/{config['data_subdirs']['psd_dir']}"
LANDCOVER_PROC_DIR = f"{config['data_dir']}/{config['data_subdirs']['landcover_proc_dir']}"
YEARS = config['years']
DECIMATION = config['decimation']
DEBUG = DECIMATION > 1

# make dirs
os.makedirs(LANDCOVER_PROC_DIR, exist_ok=True)

# ==========================================================
# read and process legend file

# legend from xlsx file
codes = pd.read_excel(f'{LANDCOVER_LEGEND_DIR}/legend.xlsx', sheet_name='legends', header=None)
codes = codes.iloc[1:, :6].reset_index(drop=True)
codes.columns = codes.iloc[0] # make header from first row
codes = codes[1:]

# make the 'NaN' column name to 'Sub-class'
codes = codes.rename(columns={codes.columns[3]: 'class-type'})
codes = codes[(codes['General class'] != 'Not used')] # remove rows with 'General class': 'Not used'
codes = codes[(codes['class-type'] != 'Not used')]  # remove rows with 'class-type': 'Not used'

# remove rows with all NaN except for 'Map value'
codes = codes.dropna(how='all', subset=codes.columns.difference(['Map value']))

# fill NaN in 'General class' with the value above
codes['General class'] = codes['General class'].ffill() 
codes['class-type'] = codes.groupby('General class')['class-type'].ffill()
codes = codes[['Map value', 'General class', 'class-type', 'Sub-class']].rename(
    columns={'Map value': 'map', 'General class': 'class', 'class-type': 'type', 'Sub-class': 'subclass'}).reset_index(drop=True)



# ==========================================================
# landcover correlations

# Lookup Tables (LUTs) for landcover features
print("Building Classification LUTs...")
df_legend = codes 

# Initialize LUTs
lut_veg_cover = np.zeros(256, dtype=np.float32) 
lut_forest    = np.zeros(256, dtype=np.float32)     
lut_wetland   = np.zeros(256, dtype=np.float32)
lut_desert    = np.zeros(256, dtype=np.float32)     
lut_urban     = np.zeros(256, dtype=np.float32)
lut_water     = np.zeros(256, dtype=np.float32)
lut_semi_arid = np.zeros(256, dtype=np.float32)
lut_crop      = np.zeros(256, dtype=np.float32)

for idx, row in df_legend.iterrows():
    try:
        code = int(row['map'])
        if code > 255: continue
        cls = str(row['class']).lower()
        typ = str(row['type']).lower()
        sub = str(row['subclass']).lower()
        
        # LUT assignments
        if "tree" in cls or "tree" in sub: lut_forest[code] = 1.0
        
        if "%" in sub and "vegetation" in sub:
            match = re.search(r'(\d+)%', sub)
            if match: lut_veg_cover[code] = float(match.group(1)) / 100.0
            
        if "wetland" in cls: lut_wetland[code] = 1.0
        if "desert" in typ: lut_desert[code] = 1.0 
        if code == 250: lut_urban[code] = 1.0
        if cls == "open surface water": lut_water[code] = 1.0
        
        if "semi-arid" in typ: lut_semi_arid[code] = 1.0
        if code == 244: 
            lut_crop[code] = 1.0        # Tracks strictly cropland
            #lut_veg_cover[code] = 1.0   # include vegetation for cropland fraction
            
    except Exception: pass

for year in YEARS:
    if os.path.exists(f"{LANDCOVER_PROC_DIR}/landcover_proc_{year}.nc"):
        print(f"Year {year} already processed..")
        continue

    # Target grid setup
    template_ds = xr.open_dataset(f"{PSD_DIR}/spectral_slopes_{year}.nc")
    lats = template_ds.latitude.values
    lons = template_ds.longitude.values
    dst_height, dst_width = len(lats), len(lons)

    res_x = abs(lons[1] - lons[0])
    res_y = abs(lats[1] - lats[0])
    west = lons.min() - (res_x / 2)
    north = lats.max() + (res_y / 2)
    dst_transform = from_origin(west, north, res_x, res_y)
    dst_crs = "EPSG:4326" 

    # Initialize Global Grids
    global_grids = {
        "fraction_veg_cover": np.full((dst_height, dst_width), np.nan, dtype=np.float32),
        "fraction_forest":    np.full((dst_height, dst_width), np.nan, dtype=np.float32),
        "fraction_wetland":   np.full((dst_height, dst_width), np.nan, dtype=np.float32),
        "fraction_desert":    np.full((dst_height, dst_width), np.nan, dtype=np.float32),
        "fraction_urban":     np.full((dst_height, dst_width), np.nan, dtype=np.float32),
        "fraction_water":     np.full((dst_height, dst_width), np.nan, dtype=np.float32),
        "fraction_semi_arid": np.full((dst_height, dst_width), np.nan, dtype=np.float32),
        "fraction_cropland":  np.full((dst_height, dst_width), np.nan, dtype=np.float32),
    }

    feature_map = {
        "fraction_veg_cover": lut_veg_cover,
        "fraction_forest":    lut_forest,
        "fraction_wetland":   lut_wetland,
        "fraction_desert":    lut_desert,
        "fraction_urban":     lut_urban,
        "fraction_water":     lut_water,
        "fraction_semi_arid": lut_semi_arid,
        "fraction_cropland":  lut_crop,
    }

    # processing each tile
    tif_files = glob.glob(f"{TIF_DIR}/landcover_{year}/*.tif")
    print(f"Processing {len(tif_files)} tiles...")

    for i, fpath in enumerate(tif_files):
        print(f"[{i+1}/{len(tif_files)}] {os.path.basename(fpath)}...")
        try:
            with rasterio.open(fpath) as src:
                
                # Decimate source data if specified
                if DECIMATION > 1:
                    new_h = src.height // DECIMATION
                    new_w = src.width // DECIMATION
                    raw_data = src.read(1, out_shape=(1, new_h, new_w), resampling=Resampling.nearest)
                    src_transform = src.transform * Affine.scale(DECIMATION, DECIMATION)
                else:
                    raw_data = src.read(1)
                    src_transform = src.transform

                # Process Variables
                for name, lut in feature_map.items():
                    
                    # map id to feature value
                    source_float = lut[raw_data]

                    # aggregate to target grid
                    temp_grid = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
                    
                    reproject(
                        source=source_float,
                        destination=temp_grid,
                        src_transform=src_transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.average, 
                        src_nodata=np.nan,
                        dst_nodata=np.nan
                    )
                    
                    # merge
                    valid_mask = np.isfinite(temp_grid)
                    global_grids[name][valid_mask] = temp_grid[valid_mask]

                del raw_data

        except Exception as e:
            print(f"Error: {e}")

    # save output
    filename = f"{LANDCOVER_PROC_DIR}/landcover_proc_{year}.nc"
    print(f"Saving to {filename}...")

    data_vars = {name: (("latitude", "longitude"), arr) for name, arr in global_grids.items()}
    ds_final = xr.Dataset(data_vars, coords={"latitude": lats, "longitude": lons})

    if os.path.exists(filename): os.remove(filename)
    ds_final.to_netcdf(filename)
    print("Done!")








