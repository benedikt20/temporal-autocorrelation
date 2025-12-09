import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import os, yaml
import xarray as xr
import Lowess_Functions as lf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# for landcover plotting
import rasterio, glob
from rasterio.enums import Resampling
import matplotlib.colors as mcolors
from collections import Counter

import geopandas as gpd
usa = gpd.read_file('https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_1_states_provinces.zip')
usa_states = usa[usa["iso_a2"] == "US"]
world = gpd.read_file('https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip')
usa_coast = world[world['NAME'] == 'United States of America']


import warnings
warnings.filterwarnings("ignore")

# ==========================================
# Config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

FIG_DIR = f"{config['figure_dir']}/{config['fig_subdirs']['pre_vis']}"
ERA5_DIR = f"{config['rawdata_dir']}/{config['rawdata_subdirs']['era5_dir']}"
PROC_LANDCOVER_DIR = f"{config['data_dir']}/{config['data_subdirs']['landcover_proc_dir']}"
RAW_LANDCOVER_DIR = f"{config['rawdata_dir']}/{config['rawdata_subdirs']['landcover_dir']}"
LANDCOVER_LEGEND_DIR = config['landcover_legend_dir']
SAVE_FIGS = config.get('save_figs', False)
DPI = config.get('dpi', 300)
SEED = config['seed']      

LAT_IDX = config['sample_point_idx']['lat']
LON_IDX = config['sample_point_idx']['lon']
YEAR = config['sample_point_idx']['year']

# decimation for landcover downsampling
DECIMATION = 20

# Create fig dir
os.makedirs(FIG_DIR, exist_ok=True)
np.random.seed(SEED)

# ==========================================
# Plot the raw data from 2020
data = xr.open_dataset(f"{ERA5_DIR}/t2m_{YEAR}.nc")

fig, ax = plt.subplots(figsize=(10, 4))
data.isel(valid_time=0).t2m.plot(ax=ax)
usa_states.boundary.plot(ax=ax, color='black', linewidth=0.4)
ax.set_title(f"2m Temperature - January 1, {YEAR} 00:00 UTC")
ax.set_aspect('equal')
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(f"{FIG_DIR}/t2m_{YEAR}_01_01_00UTC.png", dpi=DPI)
plt.close()

# =========================================================
# Plot time series at sample point
s_data = data.t2m.isel(latitude=LAT_IDX, longitude=LON_IDX).values 
plt.figure(figsize=(8, 4))
plt.plot(s_data, color='blue', alpha=0.7, lw=0.7)
plt.xlabel('Time (days)')
plt.ylabel('2m Temperature (K)')
plt.xticks(ticks=np.arange(0, len(s_data), 24*30), labels=np.arange(0, len(s_data)//24//30 + 1, 1)*30)
plt.grid(True, linestyle=':', linewidth=0.5)
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(f"{FIG_DIR}/sample_point_time_series.pdf")
plt.close()


# ==========================================
# Plot PSD processed at sample point
lx, ly, _, _ = lf.psd_proc_cell(s_data)
plt.figure(figsize=(8, 4))
plt.plot(lx, ly, color='blue', alpha=0.7, label='Multitaper PSD', lw=0.7)
plt.xlabel('Log10 Frequency')
plt.ylabel('Log10 Power')
plt.grid(True, linestyle=':', linewidth=0.5)
plt.annotate(r"$\}$",fontsize=24,
            xy=(0.6735, 0.578), xycoords='figure fraction', rotation=-90)
plt.text(0.687, 0.455, '$f_{day}$', transform=plt.gca().transAxes, fontsize=10, ha='center')
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(f"{FIG_DIR}/sample_point_psd.pdf")
plt.close()


# ==========================================
# ACF and PACF plot
# remove diurnal harmonics
s_data_detrended = pd.Series(s_data).diff(24).dropna()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
# acf
plot_acf(s_data_detrended, lags=48, ax=ax1, zero=False)
ax1.set_xticks(np.arange(0, 50, 3))
ax1.set_xlabel('Lag (hours)', fontsize=10)
ax1.set_ylabel('ACF', fontsize=10)

# pacf
plot_pacf(s_data_detrended, lags=48, ax=ax2, zero=False)
ax2.set_xticks(np.arange(0, 50, 3))
ax2.set_xlabel('Lag (hours)', fontsize=10)
ax2.set_ylabel('PACF', fontsize=10)
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(f"{FIG_DIR}/sample_point_acf_pacf.pdf")
plt.close()


# ==========================================
# Landcover plot
tifs = glob.glob(f"{RAW_LANDCOVER_DIR}/landcover_{YEAR}/*.tif")

def fix_hex(x):
    s = str(x).strip()
    return f"#{s}" if not s.startswith("#") else s

# process color legend
codes = pd.read_excel(f'{LANDCOVER_LEGEND_DIR}/legend.xlsx', sheet_name='legends', header=None)
codes = codes.iloc[1:, :6].reset_index(drop=True)
codes.columns = codes.iloc[0]
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
codes = codes[codes.columns[codes.columns.notna()]].rename(columns={'Map value': 'map', 'General class': 'class', 
                              'class-type': 'type', 'Sub-class': 'subclass', 'Color code': 'color'}).reset_index(drop=True)
codes = codes[:-1]
codes['color'] = codes['color'].apply(fix_hex)
class_ids = codes['map'].tolist()
class_colors = codes['color'].tolist()

cmap = mcolors.ListedColormap(class_colors)
boundaries = class_ids + [class_ids[-1] + 1]
norm = mcolors.BoundaryNorm(boundaries, cmap.N)

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_facecolor('#111133') 
full_extent = {'left': float('inf'), 'right': float('-inf'), 
               'bottom': float('inf'), 'top': float('-inf')}

global_counts = Counter()

for fpath in tifs:
    with rasterio.open(fpath) as src:
        new_h = src.height // DECIMATION
        new_w = src.width // DECIMATION
        if new_h == 0 or new_w == 0: continue

        data = src.read(1, out_shape=(1, new_h, new_w), resampling=Resampling.nearest)
        plot_data = data.astype(float)
        mask = np.isin(data, class_ids)
        plot_data[~mask] = np.nan 
        tile_extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

        # count landcover types
        unique, counts = np.unique(data, return_counts=True)
        global_counts.update(dict(zip(unique, counts)))
        
        ax.imshow(
            plot_data, 
            extent=tile_extent, 
            cmap=cmap, 
            norm=norm, 
            interpolation='none'
        )
        full_extent['left'] = min(full_extent['left'], src.bounds.left)
        full_extent['right'] = max(full_extent['right'], src.bounds.right)
        full_extent['bottom'] = min(full_extent['bottom'], src.bounds.bottom)
        full_extent['top'] = max(full_extent['top'], src.bounds.top)

ax.set_xlim(full_extent['left'], full_extent['right'])
ax.set_ylim(full_extent['bottom'], full_extent['top'])
ax.set_aspect('equal')
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(f"{FIG_DIR}/landcover_{YEAR}.png", dpi=DPI/2, bbox_inches='tight')
plt.close()


# ==========================================
# bar chart of landcover distribution

df_counts = pd.DataFrame.from_dict(global_counts, orient='index', columns=['count'])
df_counts.index.name = 'map'

# merge with codes to get class and type
df_merged = pd.merge(df_counts, codes, on='map', how='inner')
df_merged['type'] = df_merged['type'].fillna(df_merged['class'])

# group by class and type, sum counts
df_grouped = df_merged.groupby(['class', 'type'])['count'].sum().reset_index()

# Create a clean label column
df_grouped['label'] = df_grouped['class'] + ": " + df_grouped['type']
df_grouped = df_grouped.sort_values('count', ascending=True) 

# if same class - class, remove the second part
def simplify_label(row):
    if row['class'] == row['type']:
        return row['class']
    else:
        return row['label']
df_grouped['label'] = df_grouped.apply(simplify_label, axis=1)

# remove Ocean if present
df_grouped = df_grouped[~df_grouped['label'].str.contains('Ocean')]

plt.figure(figsize=(8, 4.5))
bars = plt.barh(df_grouped['label'], df_grouped['count'], color='navy')
plt.xlabel("Proportional pixel count")
plt.grid(axis='x', linestyle='--', alpha=0.7)

max_pct = (df_grouped['count'].max() / df_grouped['count'].sum()) * 100
pct_ticks = np.arange(0, max_pct + 5, 5)
tick_locs = (pct_ticks / 100) * df_grouped['count'].sum()
tick_labels = [f"{int(p)}%" for p in pct_ticks]
plt.xticks(ticks=tick_locs, labels=tick_labels)
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(f"{FIG_DIR}/landcover_distribution_{YEAR}.pdf")
plt.close()


# ==========================================
# 4x2 subplots
proc_landcover = xr.open_dataset(f"{PROC_LANDCOVER_DIR}/landcover_proc_{YEAR}.nc")

fig, axs = plt.subplots(2, 4, figsize=(16, 8))
for i, var in enumerate(proc_landcover.data_vars):
    ax = axs.flatten()[i]
    im = proc_landcover[var].fillna(0).plot(
        ax=ax, cmap='Blues', 
        add_colorbar=False, 
        vmin=0, vmax=1)
    
    usa_coast.boundary.plot(ax=ax, color='black', linewidth=0.4)
    ax.set_title(var)

cbar = fig.colorbar(
    im, ax=axs, 
    orientation='horizontal', 
    fraction=0.03, pad=0.1)
cbar.set_label('Fraction') 
if SAVE_FIGS:
    plt.savefig(f"{FIG_DIR}/landcover_fractions_{YEAR}.png", dpi=DPI*0.75, bbox_inches='tight')
plt.close()



# ==========================================
# Time series example
N = 2000               # Number of samples for time series
fs = 1.0               # Sampling frequency (1 sample per unit time)

def generate_colored_noise(beta, n_samples):
    """
    Generates colored noise with power law spectrum 1/f^beta.
    """
    f = np.fft.rfftfreq(n_samples)
    with np.errstate(divide='ignore'):
        scale = 1.0 / np.abs(f)**(beta / 2.0) # Scale: 1 / f^(beta/2)
    scale[0] = 0
    phases = np.random.uniform(0, 2*np.pi, len(f))
    s_freq = scale * np.exp(1j * phases)
    # Transform to time domain and normalize
    s_time = np.fft.irfft(s_freq, n=n_samples)
    s_time = (s_time - np.mean(s_time)) / np.std(s_time)
    return s_time

# Settings: 1 vs 2
beta_pink = 1.0  # Pink Noise
beta_red = 2.0   # Red Noise (Brownian)
ts_pink = generate_colored_noise(beta=beta_pink, n_samples=N)
ts_red = generate_colored_noise(beta=beta_red, n_samples=N)

# Calculate PSD
freq_pink, psd_pink = signal.welch(ts_pink, fs, nperseg=N//2)
freq_red, psd_red = signal.welch(ts_red, fs, nperseg=N//2)

# Main figure
fig, ax_main = plt.subplots(figsize=(12, 6))
ax_main.loglog(freq_pink, psd_pink, color='#d627c8', alpha=0.3)
fit_pink = 10**(np.polyval(np.polyfit(np.log10(freq_pink[1:]), np.log10(psd_pink[1:]), 1), np.log10(freq_pink[1:])))
ax_main.loglog(freq_pink[1:], fit_pink, color='#d627c8', linestyle='--', lw=2, label=r'Pink Noise (slope $ \approx -1$)')
ax_main.loglog(freq_red, psd_red, color='#d62728', alpha=0.3)
fit_red = 10**(np.polyval(np.polyfit(np.log10(freq_red[1:]), np.log10(psd_red[1:]), 1), np.log10(freq_red[1:])))
ax_main.loglog(freq_red[1:], fit_red, color='#d62728', linestyle='--', lw=2, label=r'Red Noise (slope $ \approx -2$)')
ax_main.set_xlabel("Frequency (cycles/sample)", fontsize=12)
ax_main.set_ylabel("Power", fontsize=12)
ax_main.grid(True, which="both", ls="-", alpha=0.2)
ax_main.legend(loc="upper right", fontsize=9, framealpha=0.9, edgecolor='gray')
ax_main.set_ylim([1e-4, 10**2.5])

# Inset 1: Pink Noise (Bottom Left)
ax_ins1 = fig.add_axes([0.12, 0.48, 0.22, 0.22]) 
ax_ins1.plot(ts_pink, color='#d627c8', lw=0.5, alpha=1.0)
ax_ins1.grid(True, alpha=0.3)
ax_ins1.set_ylabel("Value", fontsize=8)
ax_ins1.patch.set_alpha(0.8) 
ax_ins1.set_xlabel("Time", fontsize=8)
ax_ins1.tick_params(axis='both', which='major', labelsize=8)

# Inset 2: Red Noise (Next to Pink)
ax_ins2 = fig.add_axes([0.12, 0.175, 0.22, 0.22]) 
ax_ins2.plot(ts_red, color='#d62728', lw=1, alpha=1.0)
ax_ins2.grid(True, alpha=0.3)
ax_ins2.set_ylabel("Value", fontsize=8)
ax_ins2.set_ylim([-2, 3])
ax_ins2.patch.set_alpha(0.8)
ax_ins2.set_xlabel("Time", fontsize=8)
ax_ins2.tick_params(axis='both', which='major', labelsize=8)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/psd_slope_comparison_{beta_pink}_{beta_red}.pdf", bbox_inches='tight')
plt.close()
