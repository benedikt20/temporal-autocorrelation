import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import os, yaml
import xarray as xr
import geopandas as gpd

import warnings
warnings.filterwarnings("ignore")

# ==========================================
# Config
# ==========================================

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

FIG_DIR = f"{config['figure_dir']}/{config['fig_subdirs']['post_vis']}"
PSD_DIR = f"{config['data_dir']}/{config['data_subdirs']['psd_dir']}"
SHP_FILE = config['shp_file']
METHOD = config['method']

SAVE_FIGS = config.get('save_figs', False)
DPI = config.get('dpi', 300)
SEED = config['seed']      

# Create fig dir
os.makedirs(FIG_DIR, exist_ok=True)
np.random.seed(SEED)

# ==========================================
# beta1 and beta2 change plots
# ==========================================

# Load the shapefile
gdf = gpd.read_file(SHP_FILE).to_crs(epsg=4326)

s_data_2000 = xr.open_dataset(f"{PSD_DIR}/spectral_slopes_2000.nc")
s_data_2020 = xr.open_dataset(f"{PSD_DIR}/spectral_slopes_2020.nc")

# initialize diff dataset from s_data_2000
diff = xr.Dataset()
for var in s_data_2000.data_vars:
    diff[var] = s_data_2020[var] - s_data_2000[var]

# Plot the difference in slope1_daily
plt.figure(figsize=(10, 5))
diff[f'slope1_{METHOD}'].plot(cbar_kwargs={'label': '$\\Delta \\beta_1$'})
plt.title('Change in $\\beta_1$ (2020 - 2000)')
gdf.boundary.plot(ax=plt.gca(), color="black", linewidth=0.4, alpha=0.5)
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(f"{FIG_DIR}/beta1_change.png", dpi=DPI/2)
plt.close()

plt.figure(figsize=(10, 5))
diff[f'slope2_{METHOD}'].plot(cbar_kwargs={'label': '$\\Delta \\beta_2$'})
plt.title('Change in $\\beta_2$ (2020 - 2000)')
gdf.boundary.plot(ax=plt.gca(), color="black", linewidth=0.4, alpha=0.5)
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(f"{FIG_DIR}/beta2_change.png", dpi=DPI/2)
plt.close()



# ==================================================
# Time series example
# ==================================================

N = 2000               # Number of samples for time series
fs = 1.0               # Sampling frequency (1 sample per unit time)

# Settings: 1 vs 2
beta_pink = 1.0  # Pink Noise
beta_red = 2.0   # Red Noise (Brownian)

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
