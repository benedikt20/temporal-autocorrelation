import yaml, os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================================
# config setup
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

PSD_DIR = f"{config['data_dir']}/{config['data_subdirs']['psd_dir']}"
LANDCOVER_PROC_DIR = f"{config['data_dir']}/{config['data_subdirs']['landcover_proc_dir']}"
YEARS = config['years']
METHOD = config['method']  # 'daily' or 'br' for breakpoint
FIG_DIR = f"{config['figure_dir']}/{config['fig_subdirs']['correlations']}"
SAVE_FIGS = config.get('save_figs', False)
#DPI = config.get('dpi', 300)

os.makedirs(FIG_DIR, exist_ok=True)

# ==========================================================
# load data

# df to save the correlation matrices for each year
corrs = {}
slopes = {}

for year in YEARS:
    ds_spectral = xr.open_dataset(f"{PSD_DIR}/spectral_slopes_{year}.nc")
    ds_landcover = xr.open_dataset(f"{LANDCOVER_PROC_DIR}/landcover_proc_{year}.nc")

    # add slope difference 
    ds_spectral[f'slope_diff_{METHOD}'] = ds_spectral[f'slope1_{METHOD}'] - ds_spectral[f'slope2_{METHOD}']

    # select variables from ds_spectral that include the desired
    spectral_vars = [var for var in ds_spectral.data_vars if METHOD in var]
    land_vars = [var for var in ds_landcover.data_vars]

    # filter ds_spectral to only include vars of interest
    ds_spectral = ds_spectral[spectral_vars]

    # merge datasets
    ds = xr.merge([ds_spectral, ds_landcover]).to_dataframe().reset_index()


    # Calculate correlation between Spectral vars (rows) and Landcover vars (columns)
    corr_matrix = ds[spectral_vars + land_vars].corr().loc[spectral_vars, land_vars]
    corrs[year] = corr_matrix

    # Get stadard deviations
    std_devs = ds[spectral_vars + land_vars].std()
    slope_slice = pd.DataFrame(index=spectral_vars, columns=land_vars)
    for svar in spectral_vars:
        for lvar in land_vars:
            r = corr_matrix.loc[svar, lvar]
            std_s = std_devs[svar]
            std_l = std_devs[lvar]

            if std_l != 0:
                slope_slice.loc[svar, lvar] = r * (std_s / std_l)
            else:
                slope_slice.loc[svar, lvar] = np.nan
    slopes[year] = slope_slice

    # -----------------------------------------
    # plot correlation matrix heatmap
    # -----------------------------------------
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        corr_matrix, 
        annot=False, 
        fmt=".2f", 
        cmap='coolwarm', 
        center=0, 
        vmin=-0.6, vmax=0.6,
        cbar_kws={'label': 'Pearson Correlation (r)'}
    )

    # add text annotations
    for y in range(corr_matrix.shape[0]):
        for x in range(corr_matrix.shape[1]):
            val = corr_matrix.iloc[y, x]
            if not np.isnan(val):
                text_color = 'black'
                plt.text(
                    x + 0.5, y + 0.5, f"{val:.2f}", 
                    horizontalalignment='center', 
                    verticalalignment='center',
                    color=text_color, 
                    weight='bold',
                    fontsize=11
                )

    plt.title(f"Correlation matrix for spectral slopes vs. landcover features ({year})", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(f"{FIG_DIR}/corr_matrix_psd_landcover_{year}.pdf", bbox_inches='tight')
    plt.close()


# ==========================================================
# plot correlation time series for each spectral vs landcover variable

spectral_rows = corrs[YEARS[0]].index.tolist()
land_cols = corrs[YEARS[0]].columns.tolist()

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
axes_flat = axes.flatten()  


for i, svar in enumerate(spectral_rows[:2]):
    # Safety check: stop if we have more variables than plots (max 4)
    if i >= len(axes_flat):
        break
        
    ax = axes_flat[i]
    
    # Plot a line for every land cover variable
    for lvar in land_cols:
        r_values = [corrs[y].loc[svar, lvar] for y in YEARS]
        ax.plot(YEARS, r_values, marker='o', label=lvar)
    ax.set_title(f"{svar.replace('_daily','').replace('_br','')}", fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(0, color='black', linewidth=1, linestyle='-')
    
    # Only add Y-label to the left columns (indices 0 and 2)
    if i % 2 == 0:
        ax.set_ylabel("Pearson Correlation (r)")

# legend outside the plots
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01), frameon=False )
plt.tight_layout(rect=[0, 0.08, 1, 1])
if SAVE_FIGS:
    plt.savefig(f"{FIG_DIR}/corr_psd_landcover_years.pdf", bbox_inches='tight')
plt.close()


# ==========================================================
# Plot Regression Slopes (Physical Sensitivity)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=False) # sharey=False is safer for slopes
axes_flat = axes.flatten()  

for i, svar in enumerate(spectral_rows[:2]):
    if i >= len(axes_flat): break
    ax = axes_flat[i]
    
    for lvar in land_cols:
        # Pull from 'slopes' dictionary instead of 'corrs'
        m_values = [slopes[y].loc[svar, lvar] for y in YEARS]
        ax.plot(YEARS, m_values, marker='o', linestyle='-', label=lvar) # Changed marker to square

    ax.set_title(f"{svar.replace('_daily','').replace('_br','')}", fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(0, color='black', linewidth=1)
    
    # Y-label explanation
    if i % 2 == 0:
        #ax.set_ylabel("Regression Slope ($m$)") # \n(Change in Slope per 100% cover)
        ax.set_ylabel("Regression Slope ($m$)") # \n(Change in Spectral Slope for a unit Change in Landcover Fraction)

# Legend
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01), frameon=False)
plt.tight_layout(rect=[0, 0.08, 1, 1])

if SAVE_FIGS:
    plt.savefig(f"{FIG_DIR}/slope_matrix_years.pdf", bbox_inches='tight')
plt.close()

