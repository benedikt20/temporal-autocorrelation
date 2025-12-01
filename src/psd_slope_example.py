import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import os
import yaml

# ==========================================
# Configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
FIG_DIR = f"{config['figure_dir']}/{config['fig_subdirs']['psd_slope_example']}" 
SEED = config['seed']                            
N = 2000               # Number of samples
fs = 1.0               # Sampling frequency

# Create fig dir
os.makedirs(FIG_DIR, exist_ok=True)
np.random.seed(SEED)

# ==========================================
# Functions & Generation

def generate_colored_noise(beta, n_samples):
    """
    Generates colored noise with power law spectrum 1/f^beta.
    """
    f = np.fft.rfftfreq(n_samples)
    
    # Scale: 1 / f^(beta/2)
    with np.errstate(divide='ignore'):
        scale = 1.0 / np.abs(f)**(beta / 2.0)
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

# Generate Data
ts_pink = generate_colored_noise(beta=beta_pink, n_samples=N)
ts_red = generate_colored_noise(beta=beta_red, n_samples=N)

# Calculate PSD
freq_pink, psd_pink = signal.welch(ts_pink, fs, nperseg=N//2)
freq_red, psd_red = signal.welch(ts_red, fs, nperseg=N//2)

# ==========================================
# Plotting

# Plot 1: Pink Noise Time Series
plt.figure(figsize=(6, 4))
plt.plot(ts_pink, color='#d627c8', lw=1, alpha=0.9)
#plt.title(f"Time Series A: Pink Noise (Beta={beta_pink})", fontsize=12, weight='bold')
plt.ylabel("Amplitude")
plt.xlabel("Time")
# plt.text(0.4, 0.85, "Structured randomness\nFractal-like memory", transform=plt.gca().transAxes, 
#          bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/ts_pink_{beta_pink}.pdf")
plt.close()

# Plot 2: Red Noise Time Series
plt.figure(figsize=(6, 4))
plt.plot(ts_red, color='#d62728', lw=1.5, alpha=0.9)
#plt.title(f"Time Series B: Red Noise (Beta={beta_red})", fontsize=12, weight='bold')
plt.ylabel("Amplitude")
plt.xlabel("Time")
# plt.text(0.4, 0.8, "Strong persistence\nRandom Walk", transform=plt.gca().transAxes,
#          bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/ts_red_{beta_red}.pdf")
plt.close()

# Plot 3: PSD Comparison of Pink vs Red Noise
plt.figure(figsize=(12, 5))

# Plot Pink
plt.loglog(freq_pink, psd_pink, color='#d627c8', alpha=0.4)
fit_pink = 10**(np.polyval(np.polyfit(np.log10(freq_pink[1:]), np.log10(psd_pink[1:]), 1), np.log10(freq_pink[1:])))
plt.loglog(freq_pink[1:], fit_pink, color='#d627c8', linestyle='--', lw=2, 
           label=f'Slope $\\approx$ -1 (Pink)')

# Plot Red
plt.loglog(freq_red, psd_red, color='#d62728', alpha=0.4)
fit_red = 10**(np.polyval(np.polyfit(np.log10(freq_red[1:]), np.log10(psd_red[1:]), 1), np.log10(freq_red[1:])))
plt.loglog(freq_red[1:], fit_red, color='#d62728', linestyle='--', lw=2, 
           label=f'Slope $\\approx$ -2 (Red)')

plt.title("Power Spectral Density Comparison", fontsize=12, weight='bold')
plt.xlabel("Frequency (cycles/sample)")
plt.ylabel("Power")
plt.legend(loc='lower left', fontsize=11)
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/psd_slope_comparison_{beta_pink}_{beta_red}.pdf")
plt.close()

print(f"Plots saved to {FIG_DIR}")