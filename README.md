# temporal-autocorrelation
Title: Identifying Changes to Environmental Temporal Autocorrelation under Climate Change

--- 

### Setup and Run
Install the required packages using the following command:

```bash
pip install -r src/requirements.txt
```
Then run the main script:

```bash
python main.py
```
---

### Project Structure

```text
temporal-autocorrelation/
├── config.yaml               # Configuration of the parameters
├── main.py                   # Main script 
├── data/                     # Processed data
├── docs/                     # Documentats: report and executive summary
└── src/                     
    ├── ObtainData.py             # Downloads ERA5 temperature data and GLAD land cover data
    ├── PrepData_landcover.py     # Processes GLAD land cover data
    ├── PrepData_psd.py           # Processes ERA5 temperature data
    ├── Lowess_Functions.py       # Custom outlier detection and smoothing functions
    ├── PreVisualizations.py      # Data exploration visualizations
    ├── ModelingVisualizations.py # Visualizations for modeling section
    ├── PostVisualizations.py     # Post-processing visualizations
    └── requirements.txt          # Python dependencies
```
---

### Summary
This work investgates the temporal autocorrelation of surface level temperature across the continental United States using data from the ERA5 reanalysis dataset. The analysis also explores the effect of land cover types on the autocorrelation patterns.

We developed a non-parametric Lowess-processing approach to isolate the stochastic component from the power-density spectrum, in order to investigate the change in variability across different time scales.

Our results indicate that temperature variability is being redistributed across timescales, with long timescale temporal autocorrelation (>24 hours) decreasing across major regions such as the Pacific Northwest and New England. Conversely, short timescale autocorrelation (<24 hours) increased across most of the continental US, indicating enhanced daily-scale persistence that may contribute to extended durations of extreme events like heat waves. Additionally, the land cover analysis revealed a significant negative relationship between urbanization and low-frequency spectral exponents over time.


