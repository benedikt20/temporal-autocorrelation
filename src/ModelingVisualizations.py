import os, yaml
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import Lowess_Functions as lf
from scipy.stats import chi2
import seaborn as sns
from scipy import stats

import warnings # suppress sns density plot FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)


# =====================================================
# Config
# =====================================================
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

FIG_DIR = f"{config['figure_dir']}/{config['fig_subdirs']['model_vis']}"
ERA5_DIR = f"{config['rawdata_dir']}/{config['rawdata_subdirs']['era5_dir']}"
PSD_DIR = f"{config['data_dir']}/{config['data_subdirs']['psd_dir']}"
LANDCOVER_PROC_DIR = f"{config['data_dir']}/{config['data_subdirs']['landcover_proc_dir']}"
YEARS = config['years']
YEARS_NUM = [int(y) for y in YEARS]
METHOD = config['method']

SAVE_FIGS = config.get('save_figs', False)
DPI = config.get('dpi', 300)
#SEED = config['seed']    

LAT_IDX = config['sample_point_idx']['lat']
LON_IDX = config['sample_point_idx']['lon']
YEAR = config['sample_point_idx']['year']

K_TAPERS = config['k_tapers']    # number of tapers for MTM
CONF_RATIO = 1-10**config['conf_ratio_exp']  # confidence ratio for white noise detection

os.makedirs(FIG_DIR, exist_ok=True)

# =====================================================
# AR(1) fit for outlier detection in PSD slopes
# =====================================================

data = xr.open_dataset(f"{ERA5_DIR}/t2m_{YEAR}.nc")
s_data = data.t2m.isel(latitude=LAT_IDX, longitude=LON_IDX).values 
lx, ly, x, y = lf.psd_proc_cell(s_data)

# variance of the data and lag1 autocovariance
var = np.var(s_data)
lag1_cov = np.sum((s_data[:-1] - np.mean(s_data)) * (s_data[1:] - np.mean(s_data))) / (len(s_data) - 1)
phi = lag1_cov / var   # phi, lag-1 autocorrelation coefficient 

# variance of the white noise process
var_epsilon = var * (1 - phi**2)
#print(f"Estimated AR(1) parameters: phi={phi:.4f}, var_epsilon={var_epsilon:.4f}")

psd_ar1 = var_epsilon / (1 + phi**2 - 2 * phi * np.cos(2 * np.pi * x))

# The ratio y / psd_ar1 is distributed as chi2(v)/v
v = 2 * K_TAPERS
conf_ratio = chi2.ppf(CONF_RATIO, df=v) / v
psd_conf = psd_ar1 * conf_ratio

# Find peaks above the confidence line
peak_mask = y > psd_conf
outlier_lx = lx[peak_mask]
outlier_ly = ly[peak_mask]

# AR(1) theoretical fit and confidence curve
lx_ar1 = np.log10(x)
ly_ar1 = np.log10(psd_ar1)
ly_conf = np.log10(psd_conf)

plt.figure(figsize=(8, 4))
plt.plot(lx, ly, label='PSD', color='blue', zorder=1, lw=0.8)
plt.plot(lx_ar1, ly_ar1, label='AR(1) Fit', color='red', linestyle='--', zorder=2, lw=2)
plt.plot(lx_ar1, ly_conf, label=f'CI line', color='orange', linestyle=':', zorder=2, lw=2)
plt.scatter(outlier_lx, outlier_ly, color='magenta', label=f'Detected outliers', zorder=3, s=8)
plt.xlabel('Log10 Frequency')
plt.ylabel('Log10 Power')
plt.legend(loc = "lower left")
plt.grid(True, linestyle=':', linewidth=0.5)
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(f"{FIG_DIR}/sample_point_ar1.pdf")
plt.close()

# =====================================================
# Lowess fit for background estimation
# =====================================================

# full plot with original and distilled PSD, lowess fit, outliers
lx, ly, fig = lf.psd_lowess_proc(s_data, conf_ratio=CONF_RATIO, plot=True)
fig.tight_layout()
if SAVE_FIGS:
    fig.savefig(f"{FIG_DIR}/sample_point_lowess_proc.pdf")
plt.close()

# zoomed plot to show spectral leakage removal
lx, ly, fig = lf.psd_lowess_proc(s_data, conf_ratio=CONF_RATIO, zplot=True)
fig.tight_layout()
if SAVE_FIGS:
    fig.savefig(f"{FIG_DIR}/sample_point_lowess_proc_zoomed.pdf")
plt.close()


# =====================================================
# GLSAR fits for slope estimation
# =====================================================

*_, fig, models = lf.fit_breakpoint_glsar(lx, ly, plot=True)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(f"{FIG_DIR}/sample_point_glsar_fit.pdf")
plt.close()

br_model, lin_model = models

# save model summaries to text files
with open(f"{FIG_DIR}/sample_point_glsar_model_summaries.txt", "w") as f:
    f.write("Breakpoint Model Summary:\n")
    f.write(br_model.summary().as_text())
    f.write("\n\nLinear Model Summary:\n")
    f.write(lin_model.summary().as_text())

print(f"GLSAR summary outputs saved to txt file at {FIG_DIR}")

# =====================================================
# Density histogram of RMSE values
# =====================================================

for year in YEARS:
    year_spectral = xr.open_dataset(f"{PSD_DIR}/spectral_slopes_{year}.nc")
    if year == YEARS[0]:
        spectral = year_spectral
    else:
        spectral = xr.concat([spectral, year_spectral], dim='year')

plt.figure(figsize=(8, 4.5))
sns.kdeplot(spectral['rmse_br'].values.flatten(), color='red', fill=True, label='RMSE M1 (breakpoint)')
sns.kdeplot(spectral['rmse_daily'].values.flatten(), color='green', fill=True, label='RMSE M2 (daily breakpoint)')
sns.kdeplot(spectral['rmse_lin'].values.flatten(), color='orange', fill=True, label='RMSE M3 (no breakpoint)')
plt.legend()
plt.xlabel('RMSE', fontsize=11)
plt.ylabel('Density', fontsize=11)
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(f"{FIG_DIR}/rmse_density_histogram.pdf")
plt.close()

# =====================================================
# Breakpoint plots for all years
# =====================================================

for year in YEARS:
    year_spectral = xr.open_dataset(f"{PSD_DIR}/spectral_slopes_{year}.nc")
    if year == YEARS[0]:
        spectral = year_spectral
    else:
        spectral = xr.concat([spectral, year_spectral], dim='year')
spectral = spectral.assign_coords(year=("year", YEARS_NUM))

fig, axs = plt.subplots(2, 3, figsize=(12, 6))
for year in YEARS_NUM:
    spectral_year = spectral.sel(year=year)
    spectral_year['breakpoint'].plot(ax=axs[YEARS_NUM.index(year) // 3, YEARS_NUM.index(year) % 3], 
                                     cmap='viridis', cbar_kwargs={'label': 'Breakpoint Log-frequency [1/hour]'})
    axs[YEARS_NUM.index(year) // 3, YEARS_NUM.index(year) % 3].set_title(f"Year: {year}")
axs[1, 2].axis('off')
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(f"{FIG_DIR}/breakpoint_maps_all_years.png", dpi=DPI)
plt.close()

# =====================================================
# Slope1 and Slope2 maps for all years (fixed daily breakpoint model)
# =====================================================

# Fixed scale: get min and max
vmin_s1 = spectral['slope1_daily'].min().item()
vmax_s1 = spectral['slope1_daily'].max().item()
vmin_s2 = spectral['slope2_daily'].min().item()
vmax_s2 = spectral['slope2_daily'].max().item()

# Create subplots
fig, axs = plt.subplots(2, 5, figsize=(16, 6), layout='constrained')
for i, year in enumerate(YEARS_NUM):
    spectral_year = spectral.sel(year=year)
    im1 = spectral_year['slope1_daily'].plot(
        ax=axs[0, i], cmap='viridis', 
        vmin=vmin_s1, vmax=vmax_s1,
        add_colorbar=False)
    axs[0, i].set_title(f"$\\beta_1$: {year}")
    im2 = spectral_year['slope2_daily'].plot(
        ax=axs[1, i], cmap='viridis', 
        vmin=vmin_s2, vmax=vmax_s2,
        add_colorbar=False)
    axs[1, i].set_title(f"$\\beta_2$: {year}")

#plt.tight_layout(rect=[0, 0, 0.9, 1]) 
cbar1 = fig.colorbar(im1, ax=axs[0, :], pad=0.01, fraction=0.05, aspect=30)
cbar1.set_label('Low frequency slope $\\beta_1$')
cbar2 = fig.colorbar(im2, ax=axs[1, :], pad=0.01, fraction=0.05, aspect=30)
cbar2.set_label('High frequency slope $\\beta_2$')
if SAVE_FIGS:
    plt.savefig(f"{FIG_DIR}/slope1_slope2_maps_all_years.png", dpi=DPI)
plt.close()


# =====================================================
# Plot spectral slopes vs urban fraction with linear regression fits
# =====================================================

# setup dataframes to store results
slope_types = ['slope1', 'slope2']

fig, axs = plt.subplots(2, 5, figsize=(16, 6), sharex=True, sharey='row')
for col_idx, year in enumerate(YEARS):
    ds_spectral = xr.open_dataset(f"{PSD_DIR}/spectral_slopes_{year}.nc")
    ds_landcover = xr.open_dataset(f"{LANDCOVER_PROC_DIR}/landcover_proc_{year}.nc")
    spectral_vars = [var for var in ds_spectral.data_vars if METHOD in var]
    ds_spectral = ds_spectral[spectral_vars]
    ds_merged = xr.merge([ds_spectral, ds_landcover])
    df = ds_merged.to_dataframe().reset_index().dropna(subset=['fraction_urban', f'slope1_{METHOD}', f'slope2_{METHOD}'])

    for row_idx, slopex in enumerate(slope_types):
        ax = axs[row_idx, col_idx]
        y_col_name = f'{slopex}_{METHOD}'

        # fit linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(df['fraction_urban'], df[y_col_name])

        ax.scatter(df['fraction_urban'], df[y_col_name], alpha=0.3, s=15)
        x_vals = np.array(ax.get_xlim())
        x_vals = np.array([0, 1])
        y_vals = intercept + slope * x_vals
        ax.plot(x_vals, y_vals, color='red', lw=2, label=f'Fit: r={r_value:.2f}')

        if row_idx == 0:
            ax.set_title(f'Year: {year}', fontsize=12)
        if row_idx == 1:
            ax.set_xlabel('Urban fraction', fontsize=12)
        if col_idx == 0:
            ax.set_ylabel(f'Spectral Slope $\\beta_{row_idx+1}$', fontsize=12)

        # add statistics box
        stats_text = f'$\\gamma={slope:.2f}$\n$\\rho={r_value:.2f}$'
        ax.text(0.95, 0.05, stats_text,
                transform=ax.transAxes, fontsize=12,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(f"{FIG_DIR}/spectral_slopes_vs_urban_fraction.pdf")
plt.close()