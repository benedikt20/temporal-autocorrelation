import yaml, os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats 

# ==========================================================
# config setup
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

PSD_DIR = f"{config['data_dir']}/{config['data_subdirs']['psd_dir']}"
LANDCOVER_PROC_DIR = f"{config['data_dir']}/{config['data_subdirs']['landcover_proc_dir']}"
YEARS = config['years']
METHOD = config['method']
FIG_DIR = f"{config['figure_dir']}/{config['fig_subdirs']['correlations']}"
SAVE_FIGS = config.get('save_figs', False)
SEED = config.get('seed', 42)
BLOCK_SIZE = config.get('block_size', 10)
N_BOOTSTRAP = config.get('n_bootstrap', 1000)

os.makedirs(FIG_DIR, exist_ok=True)
np.random.seed(SEED)


# ==========================================================
# BOOTSTRAP FUNCTION
# ==========================================================
def spatial_block_bootstrap(df, x_col, y_col, block_col, n_boots=N_BOOTSTRAP): 
    """
    Performs block bootstrapping to estimate SE of slope and correlation.
    """
    blocks = df[block_col].unique()
    n_blocks = len(blocks)
    
    boot_slopes = []
    boot_corrs = []
    
    grouped = dict(list(df.groupby(block_col)))
    
    for i in range(n_boots):
        # 1. Resample Blocks with replacement
        resampled_block_ids = np.random.choice(blocks, size=n_blocks, replace=True)
        
        # 2. Reconstruct the dataset
        boot_chunks = [grouped[bid] for bid in resampled_block_ids]
        boot_df = pd.concat(boot_chunks)
        
        # 3. Calculate Stats
        if len(boot_df) > 10:
            res = stats.linregress(boot_df[x_col], boot_df[y_col])
            boot_slopes.append(res.slope)
            boot_corrs.append(res.rvalue)
            
    # Calculate Standard Error (Std Dev of the bootstrap distribution)
    se_slope = np.std(boot_slopes)
    se_corr = np.std(boot_corrs)
    
    return se_slope, se_corr


# ==========================================================
# MAIN LOOP
# ==========================================================

corrs = {}
slopes = {}
corr_errs = {}
slope_errs = {}

for year in YEARS:
    print(f"Processing {year} with Block Bootstrapping...")
    ds_spectral = xr.open_dataset(f"{PSD_DIR}/spectral_slopes_{year}.nc")
    ds_landcover = xr.open_dataset(f"{LANDCOVER_PROC_DIR}/landcover_proc_{year}.nc")

    # add slope difference 
    ds_spectral[f'slope_diff_{METHOD}'] = ds_spectral[f'slope1_{METHOD}'] - ds_spectral[f'slope2_{METHOD}']

    spectral_vars = [var for var in ds_spectral.data_vars if METHOD in var]
    land_vars = [var for var in ds_landcover.data_vars]
    ds_spectral = ds_spectral[spectral_vars]

    # ---------------------------------------------------------
    # 1. Assign Block IDs (Grid Tiling)
    # ---------------------------------------------------------
    ds_merged = xr.merge([ds_spectral, ds_landcover])
    
    lat_idx = np.arange(ds_merged.latitude.size)
    lon_idx = np.arange(ds_merged.longitude.size)
    lat_grid, lon_grid = np.meshgrid(lat_idx, lon_idx, indexing='ij')
    
    # Create simple block ID strings "row_col"
    block_id_flat = [f"{r // BLOCK_SIZE}_{c // BLOCK_SIZE}" for r, c in zip(lat_grid.ravel(), lon_grid.ravel())]
    
    df = ds_merged.to_dataframe().reset_index()
    df['block_id'] = block_id_flat
    
    # Drop NaNs now to speed up bootstrap
    df = df.dropna()

    # Initialize matrices
    c_mat = pd.DataFrame(index=spectral_vars, columns=land_vars, dtype=float)
    c_err_mat = pd.DataFrame(index=spectral_vars, columns=land_vars, dtype=float)
    s_mat = pd.DataFrame(index=spectral_vars, columns=land_vars, dtype=float)
    s_err_mat = pd.DataFrame(index=spectral_vars, columns=land_vars, dtype=float)

    # ---------------------------------------------------------
    # 2. Run Bootstrap Loop
    # ---------------------------------------------------------
    for svar in spectral_vars:
        for lvar in land_vars:
            pair_df = df[[svar, lvar, 'block_id']]
            
            if len(pair_df) > 10:
                # Point Estimate
                res_orig = stats.linregress(pair_df[lvar], pair_df[svar])
                orig_slope = res_orig.slope
                orig_corr = res_orig.rvalue
                
                # Bootstrap Error Estimate
                se_slope, se_corr = spatial_block_bootstrap(
                    pair_df, x_col=lvar, y_col=svar, block_col='block_id', n_boots=N_BOOTSTRAP
                )
                
                s_mat.loc[svar, lvar] = orig_slope
                s_err_mat.loc[svar, lvar] = se_slope
                
                c_mat.loc[svar, lvar] = orig_corr
                c_err_mat.loc[svar, lvar] = se_corr
            else:
                s_mat.loc[svar, lvar] = np.nan
                # ... fill others with nan

    corrs[year] = c_mat
    corr_errs[year] = c_err_mat
    slopes[year] = s_mat
    slope_errs[year] = s_err_mat

    # -----------------------------------------
    # plot correlation matrix heatmap
    # -----------------------------------------
    plt.figure(figsize=(12, 8))
    
    # FIX: Use 'c_mat' (calculated above), NOT 'corr_matrix'
    sns.heatmap(
        c_mat, 
        annot=False, 
        fmt=".2f", 
        cmap='coolwarm', 
        center=0, 
        vmin=-0.6, vmax=0.6,
        cbar_kws={'label': 'Pearson Correlation (r)'}
    )

    for y in range(c_mat.shape[0]):
        for x in range(c_mat.shape[1]):
            val = c_mat.iloc[y, x]
            if not np.isnan(val):
                plt.text(
                    x + 0.5, y + 0.5, f"{val:.2f}", 
                    horizontalalignment='center', verticalalignment='center',
                    color='black', weight='bold', fontsize=11
                )

    plt.title(f"Correlation matrix ({year})", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(f"{FIG_DIR}/corr_matrix_psd_landcover_{year}.pdf", bbox_inches='tight')
    plt.close()


# ==========================================================
# plot correlation time series (with error bars)

spectral_rows = corrs[YEARS[0]].index.tolist()
land_cols = corrs[YEARS[0]].columns.tolist()

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
axes_flat = axes.flatten()  


for i, svar in enumerate(spectral_rows[:2]):
    if i >= len(axes_flat): break
    ax = axes_flat[i]
    
    for lvar in land_cols:
        y_vals = [corrs[y].loc[svar, lvar] for y in YEARS]
        y_errs = [corr_errs[y].loc[svar, lvar] for y in YEARS] # <--- GET ERRORS
        
        # FIX: Use errorbar instead of plot
        ax.errorbar(
            YEARS, y_vals, yerr=y_errs, 
            marker='o', capsize=3, elinewidth=1.5,
            label=lvar, alpha=0.8
        )
        
    ax.set_title(f"{svar.replace('_daily','').replace('_br','')}", fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(0, color='black', linewidth=1, linestyle='-')
    
    if i % 2 == 0:
        ax.set_ylabel("Pearson Correlation (r)")

handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01), frameon=False )
plt.tight_layout(rect=[0, 0.08, 1, 1])
if SAVE_FIGS:
    plt.savefig(f"{FIG_DIR}/corr_psd_landcover_years.pdf", bbox_inches='tight')
plt.close()


# ==========================================================
# Plot Regression Slopes (with error bars)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=False)
axes_flat = axes.flatten()  

for i, svar in enumerate(spectral_rows[:2]):
    if i >= len(axes_flat): break
    ax = axes_flat[i]
    
    for lvar in land_cols:
        m_vals = [slopes[y].loc[svar, lvar] for y in YEARS]
        m_errs = [slope_errs[y].loc[svar, lvar] for y in YEARS] # <--- GET ERRORS
        
        # errorbar plot
        ax.errorbar(
            YEARS, m_vals, yerr=m_errs,
            marker='s', linestyle='-', capsize=3, elinewidth=1.5,
            label=lvar, alpha=0.8
        )

    ax.set_title(f"{svar.replace('_daily','').replace('_br','')}", fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(0, color='black', linewidth=1)
    
    if i % 2 == 0:
        ax.set_ylabel("Regression Slope ($m$)")

handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01), frameon=False)
plt.tight_layout(rect=[0, 0.08, 1, 1])

if SAVE_FIGS:
    plt.savefig(f"{FIG_DIR}/slope_matrix_years.pdf", bbox_inches='tight')
plt.close()