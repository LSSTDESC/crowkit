#from crowkit import ClusterCountsForecast
import crowkit
import os
# Single run
import numpy as np

forecast = crowkit.ClusterCountsForecast("cluster_config.yaml")

# --- Stage 1: check yaml loaded correctly ---
print("=== CONFIG CHECK ===")
print(f"active_params : {forecast.active_params}")
print(f"Omega_m       : {forecast._cosmo['Omega_m']}")
print(f"sigma8        : {forecast._cosmo['sigma8']}")
print(f"sky_area      : {forecast._survey['sky_area']}")
print(f"z_bins        : {forecast._binning['z_min']} -> {forecast._binning['z_max']} ({forecast._binning['n_z_bins']} bins)")
print(f"proxy_bins    : {forecast._binning['proxy_min']} -> {forecast._binning['proxy_max']} ({forecast._binning['n_proxy_bins']} bins)")
print(f"selection_fn  : {forecast._cfg['counts_modeling']['selection_function']}")

# --- Stage 2: check theta0 builds ---
print("\n=== THETA0 ===")
forecast.theta0 = forecast._build_theta0()
print(f"theta0        : {forecast.theta0}")

# --- Stage 3: check bins build ---
print("\n=== BINS ===")
b = forecast._binning
forecast.z_bins     = forecast._make_bin_edges(b["z_min"], b["z_max"], b["n_z_bins"])
forecast.proxy_bins = forecast._make_bin_edges(b["proxy_min"], b["proxy_max"], b["n_proxy_bins"])
print(f"z_bins     : {forecast.z_bins}")
print(f"proxy_bins : {forecast.proxy_bins}")

# --- Stage 4: single model call at fiducial ---
print("\n=== SINGLE MODEL CALL (this is the slow part) ===")
import time
forecast._model_fn, _, _ = forecast._build_model_fn()
t0 = time.time()
counts = forecast._model_fn(forecast.theta0)
print(f"counts      : {counts}")
print(f"time        : {time.time() - t0:.1f}s")
print(f"n_bins      : {len(counts)}  (expected {forecast._binning['n_z_bins'] * forecast._binning['n_proxy_bins']})")

# --- Stage 5: covariance (tjpcov — second slow part) ---
print("\n=== COVARIANCE (tjpcov) ===")
forecast._counts_fid = counts
t0 = time.time()
cov = forecast._build_covariance()
print(f"cov shape   : {cov.shape}")
print(f"cov diag    : {np.diag(cov)}")
print(f"time        : {time.time() - t0:.1f}s")


forecast.run().print_summary()
forecast.plot()

#Sweep (reads sweep section from yaml)
forecasts = forecast.run_sweep("cluster_config.yaml")
