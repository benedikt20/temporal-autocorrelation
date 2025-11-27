import numpy as np
import matplotlib.pyplot as plt
import spectrum
import scipy
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import chi2

# model fitting imports
from scipy.stats import binned_statistic
import statsmodels.api as sm

# suppress runtime warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ======================================
# LOWESS-related functions for PSD processing
# ======================================

def distill_outliers(lx, ly, peak_mask, avg_neighbors=5):
    ly_distilled = ly.copy()
    outlier_lx = lx[peak_mask]

    for i, outlier in enumerate(outlier_lx):
        outlier_idx = np.where(lx == outlier)[0][0]
        # find closest 5 lower and higher non-outlier points
        lower_indices = []
        higher_indices = []

        # while left
        offset = 1
        while len(lower_indices) < avg_neighbors and outlier_idx - offset >= 0:
            if not peak_mask[outlier_idx - offset]:
                lower_indices.append(outlier_idx - offset)
            offset += 1
        # while right
        offset = 1
        while len(higher_indices) < avg_neighbors and outlier_idx + offset < len(lx):
            if not peak_mask[outlier_idx + offset]:
                higher_indices.append(outlier_idx + offset)
            offset += 1

        # take the minimum number of neighbors found
        min_neighbors = min(len(lower_indices), len(higher_indices))
        lower_indices = lower_indices[:min_neighbors]
        higher_indices = higher_indices[:min_neighbors]

        # calculate average power
        neighbor_indices = lower_indices + higher_indices
        avg_power = np.mean(ly[neighbor_indices])
        # set outlier power to average for ly
        ly_distilled[np.where(lx == outlier)[0][0]] = avg_power

    return ly_distilled

def remove_spectral_leakage(lx, ly, wiggly_frac=0.05, stiff_frac=0.8):
    lx_new = lx.copy()
    ly_new = ly.copy()
    
    if not np.all(np.isfinite(ly)):
        finite_mask = np.isfinite(ly)
        lx_new = lx[finite_mask]
        ly_new = ly[finite_mask]

    # diff between smooth2 and smooth1 as relu
    wiggly = lowess(ly, lx, frac=wiggly_frac)[:, 1]
    stiff = lowess(ly, lx, frac=stiff_frac)[:, 1]
    diff = np.maximum(wiggly - stiff, 0)
    # subtract diff from ly
    ly_new -= diff

    return lx_new, ly_new, wiggly, stiff


# ======================================
# Main LOWESS processing function
# ======================================


def psd_lowess_proc(data_1d, smooth_frac=0.3, conf_ratio=1-1e-9, k_tapers=3, bandwidth=2, plot=False):
    """
    Plots the MTM PSD and fits a LOWESS-smoothed background.
          - smooth_frac: higher values = smoother fit
    """
    d_in = scipy.signal.detrend(data_1d, type='linear') # detrend
    psd = spectrum.mtm.MultiTapering(d_in, NW=bandwidth, k=k_tapers)

    x = np.array(psd.frequencies())
    y = np.array(psd.psd)
    
    # filter out non-positive and non-finite values
    valid_mask = (x > 0) & np.isfinite(y)
    x = x[valid_mask]
    y = y[valid_mask]
    
    lx = np.log10(x)
    ly = np.log10(y)

    # compute LOWESS fit: returns [x_sorted, y_smoothed]
    smoothed = lowess(ly, lx, frac=smooth_frac)
    lx_smooth = smoothed[:, 0]
    ly_smooth = smoothed[:, 1]

    v = 2 * k_tapers
    conf_ratio_curve = chi2.ppf(conf_ratio, df=v) / v
    conf_offset = np.log10(conf_ratio_curve)

    # confidence line is LOWESS fit + offset
    ly_conf = ly_smooth + conf_offset
    
    # identify peaks above confidence line
    peak_mask = ly > ly_conf
    outlier_lx = lx[peak_mask]
    outlier_ly = ly[peak_mask]

    # filter outliers with means and quadratic subtraction
    ly_distilled = distill_outliers(lx, ly, peak_mask)

    lx_new, ly_new, wiggly, stiff = remove_spectral_leakage(lx, ly_distilled, wiggly_frac=0.05, stiff_frac=0.8)


    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(lx, ly, label='PSD (original)', color='cyan', zorder=1, lw=0.8) # original PSD
        plt.plot(lx_new, ly_new, label='PSD_quad_adj', color='blue', zorder=1, lw=0.8) # after quadratic adjustment

        plt.plot(lx_new, wiggly, label='Wiggly (frac=0.05)', color='orange', linestyle='--', zorder=2, lw=1)
        plt.plot(lx_new, stiff, label='Stiff (frac=0.3)', color='purple', linestyle='--', zorder=2, lw=1)

        # Plot the LOWESS fit and confidence line, and outliers
        plt.plot(lx_smooth, ly_smooth, label=f'LOWESS Fit (frac={smooth_frac})', color='red', linestyle='--', zorder=2, lw=2)
        plt.plot(lx_smooth, ly_conf, label=f'Significance Threshold ({conf_ratio} CI)', color='orange', linestyle=':', zorder=2, lw=2)
        plt.scatter(outlier_lx, outlier_ly, color='magenta', label=f'Significant Peaks', zorder=3, s=10)

        plt.xlabel('Log10(Frequency)')
        plt.ylabel('Log10(Power)')
        plt.title('Multitaper PSD with LOWESS Background Fit')
        plt.legend(loc="lower left")
        plt.grid(True, linestyle=':', linewidth=0.5)
        plt.show()

    return lx_new, ly_new 


def log_binning(x, y, num_bins=100):
    log_bins = np.linspace(x.min(), x.max(), num_bins + 1) # log-binning

    # bin the ly values according to lx bins, using the means of each bin
    bin_means_y, bin_edges, binnumber = binned_statistic(x, y, statistic='mean', bins=log_bins)
    bin_centers_x = (bin_edges[:-1] + bin_edges[1:]) / 2

    # remove empty bins (where bin_means_y is NaN)
    nan_mask = np.isnan(bin_means_y)
    binned_x = bin_centers_x[~nan_mask]
    binned_y = bin_means_y[~nan_mask]
    
    return binned_x, binned_y


def fit_breakpoint_glsar(lx, ly, Nbin=100, plot=False):
    binned_x, binned_y = log_binning(lx, ly, num_bins=Nbin)

    best_aic = np.inf
    best_model = None
    best_breakpoint = None

    # set search range for breakpoints
    min_idx, max_idx = int(len(binned_x) * 0.1), int(len(binned_x) * 0.9)

    for i in range(min_idx, max_idx):
        breakpoint_val = binned_x[i]
        
        # design matrix: 
        # first slope: x1 = x
        # second slope: hinge function max(0, x - breakpoint)
        x1 = binned_x
        x2 = np.maximum(0, binned_x - breakpoint_val)
        X_design = sm.add_constant(np.column_stack((x1, x2)))

        # Fit GLSAR model with AR(1) errors
        model_gls = sm.GLSAR(binned_y, X_design, rho=1)
        results_gls = model_gls.fit()

        # store best model based on AIC
        if results_gls.aic < best_aic:
            best_aic = results_gls.aic
            best_model = results_gls
            best_breakpoint = breakpoint_val

    params = best_model.params # [intercept, slope1, slope_change (slope2 - slope1)]
    intercept = params[0]
    slope1_br = params[1]
    slope2_br = slope1_br + params[2]
    rmse_br = np.sqrt(np.mean(best_model.resid**2)) # RMSE of best model

    # fixed breakpoint at daily freq for RMSE comparison
    fixed_bp = np.log10(1/24)   # daily frequency in log10(Hz)
    x1 = binned_x
    x2 = np.maximum(0, binned_x - fixed_bp)
    X_fixed = sm.add_constant(np.column_stack((x1, x2)))
    model_fixed = sm.GLSAR(binned_y, X_fixed, rho=1)
    res_fixed = model_fixed.fit()
    slope1_dailybr = res_fixed.params[1]
    slope2_dailybr = slope1_dailybr + res_fixed.params[2]
    rmse_dailybr = np.sqrt(np.mean(res_fixed.resid**2))

    # no breakpoint linear model for RMSE comparison
    X_lin = sm.add_constant(binned_x)   # intercept + slope
    model_lin = sm.GLSAR(binned_y, X_lin, rho=1)
    res_lin = model_lin.fit()
    slope_lin = res_lin.params[1]
    rmse_lin = np.sqrt(np.mean(res_lin.resid**2))

    if plot:
        plt.figure(figsize=(12, 7))
        plt.plot(lx, ly, color='blue', alpha=0.3, label='Original Processed PSD')
        plt.scatter(binned_x, binned_y, color='blue', label='Binned Data (Means)', s=20, zorder=3)
        plt.plot(binned_x, best_model.fittedvalues, color='red', linestyle='--', linewidth=1.5, label='Piecewise GLSAR Fit', zorder=4)
        # plt.plot(binned_x, res_fixed.fittedvalues, color='green', linestyle=':', linewidth=1.5, label='Fixed Breakpoint (Daily Freq)', zorder=2)
        plt.plot(binned_x, res_lin.fittedvalues, color='orange', linestyle='-.', linewidth=1.5, 
                 label='Linear GLSAR Fit (No Breakpoint)', zorder=2)
        plt.axvline(x=best_breakpoint, color='black', linestyle=':', linewidth=2, 
                    label=f'Breakpoint = {best_breakpoint:.4f}')
        # add slope annotations
        plt.text(0.1, 0.8, f'Slope 1: {slope1_br:.4f}', 
                transform=plt.gca().transAxes, fontsize=12, color='darkred')
        plt.text(0.7, 0.3, f'Slope 2: {slope2_br:.4f}', 
                transform=plt.gca().transAxes, fontsize=12, color='darkred')

        plt.xlabel('Log10(Frequency)')
        plt.ylabel('Log10(Power)')
        plt.title('Piecewise GLSAR Fit of Power Spectrum')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.show()

    return best_breakpoint, slope1_br, slope2_br, rmse_br, slope1_dailybr, slope2_dailybr, rmse_dailybr, slope_lin, rmse_lin