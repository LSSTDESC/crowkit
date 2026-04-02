"""
Cluster-count forecast using CROW and DerivKit, unified into a single class.

Usage — single forecast:
    forecast = ClusterCountsForecast("config.yaml")
    forecast.run()
    forecast.plot()

Usage — sweep:
    forecasts = ClusterCountsForecast.run_sweep("config.yaml")

The sweep is defined inside the yaml under the 'sweep' key.
Individual sweep cases can also be passed at runtime:
    forecasts = ClusterCountsForecast.run_sweep(
        "config.yaml",
        sweep={"sky_area": [440, 1000, 5000]},
    )
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import copy
import itertools
from pathlib import Path
from typing import Optional, Union

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
import yaml
from derivkit import ForecastKit
from getdist import plots as getdist_plots

from crow.cluster_modules.completeness_models import CompletenessAguena16
from crow.cluster_modules.purity_models import PurityAguena16
from crow.cluster_modules.kernel import SpectroscopicRedshift
from crow.cluster_modules.mass_proxy import MurataUnbinned
from crow.cluster_modules.abundance import ClusterAbundance
from crow.recipes.binned_grid import GridBinnedClusterRecipe


import tempfile
import sacc
from tjpcov.covariance_calculator import CovarianceCalculator

# GetDist compatibility for older numpy
if not hasattr(np, "infty"):
    np.infty = np.inf

# ---------------------------------------------------------------------------
# Parameter metadata
# ---------------------------------------------------------------------------

PARAM_LABELS = {
    "Omega_m": r"\Omega_m",
    "sigma8": r"\sigma_8",
    "mu0": r"\mu_0",
    "mu1": r"\mu_1",
    "mu2": r"\mu_2",
    "sigma0": r"\sigma_0",
    "sigma1": r"\sigma_1",
    "sigma2": r"\sigma_2",
}

# Maps every sweepable / forecast param to the yaml section that owns it.
# Used by _case_to_overrides to convert a flat {param: value} dict into
# the nested yaml structure that _deep_merge expects.
PARAM_SECTION = {
    "Omega_m": "cosmology",
    "Omega_b": "cosmology",
    "h": "cosmology",
    "n_s": "cosmology",
    "sigma8": "cosmology",
    "pivot_redshift": "mass_observable",
    "pivot_mass": "mass_observable",
    "mu0": "mass_observable",
    "mu1": "mass_observable",
    "mu2": "mass_observable",
    "sigma0": "mass_observable",
    "sigma1": "mass_observable",
    "sigma2": "mass_observable",
    "sky_area": "survey",
    "mass_min": "counts_modeling",
    "mass_max": "counts_modeling",
    "selection_function": "counts_modeling",
    "completeness_a_n": "counts_modeling",
    "completeness_b_n": "counts_modeling",
    "completeness_a_logm_piv": "counts_modeling",
    "completeness_b_logm_piv": "counts_modeling",
    "purity_a_n": "counts_modeling",
    "purity_b_n": "counts_modeling",
    "purity_a_logm_piv": "counts_modeling",
    "purity_b_logm_piv": "counts_modeling",
    "photoz_err": "counts_modeling",
    "z_min": "binning",
    "z_max": "binning",
    "n_z_bins": "binning",
    "proxy_min": "binning",
    "proxy_max": "binning",
    "n_proxy_bins": "binning",
    "mass_grid_size": "grid",
    "redshift_grid_size": "grid",
    "proxy_grid_size": "grid",
    "method": "deriv",
    "stepsize": "deriv",
    "num_points": "deriv",
    "extrapolation": "deriv",
    "levels": "deriv",
    "forecast_order": "deriv",
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ClusterCountsForecast:

    PARAM_LABELS = PARAM_LABELS

    def __init__(self, yaml_path: Union[str, Path]):
        self._cfg = self._load_yaml(yaml_path)
        self._yaml_path = Path(yaml_path)

        # convenience accessors into the nested dict
        self._cosmo   = self._cfg["cosmology"]
        self._mor     = self._cfg["mass_observable"]
        self._survey  = self._cfg["survey"]
        self._binning = self._cfg["binning"]
        self._grid    = self._cfg["grid"]
        self._deriv   = self._cfg["deriv"]
        self._fcast   = self._cfg["forecast"]
        self._priors  = self._cfg.get("priors", {})
        self._plot    = self._cfg.get("plot", {})

        self.active_params: list = list(self._fcast["active_params"])

        # populated by setup()
        self.theta0: np.ndarray = None
        self.z_bins: list = None
        self.proxy_bins: list = None
        self._model_fn = None
        self._cov: np.ndarray = None
        self._counts_fid: np.ndarray = None
        self._sigma_map: dict = {}
        self._fk = None

        # results populated by run()
        self.fisher_matrix: np.ndarray = None
        self.fisher_posterior: np.ndarray = None
        self.dali_result = None
        self._getdist_objects: list = []
        self._legend_labels: list = []
        self._contour_colors: list = []

        self._is_setup = False

    # ------------------------------------------------------------------
    # Yaml helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_yaml(path: Union[str, Path]) -> dict:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cfg

    @staticmethod
    def _deep_merge(base: dict, overrides: dict) -> dict:
        """
        Recursively merge overrides into a deep copy of base.
        Scalar values in overrides replace those in base.
        Nested dicts are merged recursively.
        """
        result = copy.deepcopy(base)
        for key, value in overrides.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = ClusterCountsForecast._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    @staticmethod
    def _case_to_overrides(case: dict) -> dict:
        """
        Convert a flat {param: value} case dict into a nested override dict.

        Example:
            {"sky_area": 1000, "sigma0": 0.2}
            ->
            {"survey": {"sky_area": 1000}, "mass_observable": {"sigma0": 0.2}}
        """
        overrides = {}
        for param, value in case.items():
            section = PARAM_SECTION.get(param)
            if section is None:
                raise ValueError(
                    f"Sweep parameter '{param}' is not listed in PARAM_SECTION. "
                    f"Add it manually if needed."
                )
            overrides.setdefault(section, {})[param] = value
        return overrides

    @staticmethod
    def _cartesian_cases(sweep: dict) -> list:
        """Cartesian product of sweep values -> list of {param: value} dicts."""
        keys = list(sweep.keys())
        value_lists = [sweep[k] for k in keys]
        return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]

    @staticmethod
    def _format_case_label(case: dict) -> str:
        return ", ".join(f"{k}={v}" for k, v in case.items())

    # ------------------------------------------------------------------
    # Config accessors
    # ------------------------------------------------------------------

    def _fiducial_value(self, param: str) -> float:
        """Look up the fiducial value of a param across all config sections."""
        for section in ("cosmology", "mass_observable", "survey", "binning", "counts_modeling"):
            if param in self._cfg.get(section, {}):
                return float(self._cfg[section][param])
        raise KeyError(f"Parameter '{param}' not found in any config section.")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _resolve_params(self, theta: np.ndarray) -> dict:
        """
        Build a full param dict from fiducial values, then overwrite
        the active entries with the values in theta.
        """
        params = {}
        for section in ("cosmology", "mass_observable"):
            params.update(self._cfg[section])
        for name, value in zip(self.active_params, theta):
            params[name] = float(value)
        return params

    @staticmethod
    def _make_bin_edges(vmin: float, vmax: float, n_bins: int) -> list:
        edges = np.linspace(vmin, vmax, n_bins + 1)
        return [(float(edges[i]), float(edges[i + 1])) for i in range(n_bins)]

    def _build_theta0(self) -> np.ndarray:
        return np.array(
            [self._fiducial_value(p) for p in self.active_params], dtype=float
        )

    def _build_murata(self, p: dict) -> MurataUnbinned:
        md = MurataUnbinned(
            pivot_log_mass=p["pivot_mass"],
            pivot_redshift=p["pivot_redshift"],
        )
        for key in ("mu0", "mu1", "mu2", "sigma0", "sigma1", "sigma2"):
            md.parameters[key] = float(p[key])
        return md

    def _build_model_fn(self):
        """
        Return a callable model(theta) -> np.ndarray of cluster counts.
        Captures config references but not mutable state.
        """
        cosmo_cfg   = self._cosmo
        survey_cfg  = self._survey
        binning_cfg = self._binning
        grid_cfg    = self._grid
        active      = self.active_params
        cfg         = self._cfg
        build_murata = self._build_murata

        z_bins     = self._make_bin_edges(
            binning_cfg["z_min"], binning_cfg["z_max"], binning_cfg["n_z_bins"]
        )
        proxy_bins = self._make_bin_edges(
            binning_cfg["proxy_min"], binning_cfg["proxy_max"], binning_cfg["n_proxy_bins"]
        )

        def model(theta):
            # build full param dict
            params = {}
            for section in ("cosmology", "mass_observable", "counts_modeling"):
                params.update(cfg[section])
            for name, value in zip(active, theta):
                params[name] = float(value)

            Omega_c = params["Omega_m"] - cosmo_cfg["Omega_b"]
            if Omega_c <= 0:
                raise ValueError("Omega_m must be greater than Omega_b.")

            cosmo = ccl.Cosmology(
                Omega_c=Omega_c,
                Omega_b=cosmo_cfg["Omega_b"],
                h=cosmo_cfg["h"],
                sigma8=params["sigma8"],
                n_s=cosmo_cfg["n_s"],
            )
            mass_distribution = build_murata(params)
            hmf = ccl.halos.MassFuncDespali16(mass_def="200c")
            cluster_theory = ClusterAbundance(cosmo, hmf)
            if params["selection_function"]:
                comp = CompletenessAguena16()
                pur = PurityAguena16()
                prefix_comp = "completeness_"
                prefix_pur = "purity_"
                for key in ("a_n", "b_n", "a_logm_piv", "b_logm_piv"):
                    comp.parameters[key] = float(params[f"{prefix_comp}{key}"])            
                    pur.parameters[key] =  float(params[f"{prefix_pur}{key}"])
            else:
                comp = None
                pur = None
            recipe = GridBinnedClusterRecipe(
                mass_interval=(params["mass_min"], params["mass_max"]),
                cluster_theory=cluster_theory,
                redshift_distribution=SpectroscopicRedshift(),
                mass_distribution=mass_distribution,
                completeness=comp,
                purity=pur,
                proxy_grid_size=grid_cfg["proxy_grid_size"],
                redshift_grid_size=grid_cfg["redshift_grid_size"],
                mass_grid_size=grid_cfg["mass_grid_size"],
            )
            recipe.setup()

            counts_list = []
            for z_bin in z_bins:
                for proxy_bin in proxy_bins:
                    counts = recipe.evaluate_theory_prediction_counts(
                        z_bin, proxy_bin, survey_cfg["sky_area"]
                    )
                    counts_list.append(counts)

            return np.array(counts_list, dtype=float)

        return model, z_bins, proxy_bins

    def _build_covariance(self) -> np.ndarray:
        """
        Build the covariance matrix using tjpcov (Gaussian + SSC).
        Internally creates a minimal sacc file from the current binning,
        runs CovarianceCalculator, extracts the numpy block, and returns it.
        The sacc file is temporary and not exposed to the user.
        """
        survey_name = "survey"
        s = sacc.Sacc()
        s.add_tracer("survey", survey_name, self._survey["sky_area"])

        bin_z_labels = []
        bin_richness_labels = []

        for i, (z_low, z_high) in enumerate(self.z_bins):
            label = f"bin_z_{i}"
            s.add_tracer("bin_z", label, z_low, z_high)
            bin_z_labels.append(label)

        for i, (p_low, p_high) in enumerate(self.proxy_bins):
            label = f"bin_richness_{i}"
            s.add_tracer(
                "bin_richness",
                label,
                np.log10(p_low),   # tjpcov expects log10(proxy)
                np.log10(p_high),
            )
            bin_richness_labels.append(label)

        # populate data points with fiducial counts so tjpcov has
        # something to work with — values will be overwritten by the
        # covariance computation anyway
        cluster_count_type = sacc.standard_types.cluster_counts
        for count, (z_label, r_label) in zip(
            self._counts_fid,
            itertools.product(bin_z_labels, bin_richness_labels),
        ):
            s.add_data_point(
                cluster_count_type,
                (survey_name, r_label, z_label),
                float(count),
            )

        # diagonal placeholder so sacc is valid before tjpcov overwrites it
        s.add_covariance(np.diag(np.maximum(self._counts_fid, 1.0)))
        s.to_canonical_order()

        with tempfile.NamedTemporaryFile(
            suffix=".sacc", delete=False
        ) as tmp:
            tmp_sacc_path = tmp.name

        try:
            s.save_fits(tmp_sacc_path, overwrite=True)
            tjpcov_cfg = self._build_tjpcov_config(tmp_sacc_path)
            cc = CovarianceCalculator(tjpcov_cfg)
            cov = cc.get_covariance()
        finally:
            os.unlink(tmp_sacc_path)

        return np.array(cov, dtype=float)


    def _build_tjpcov_config(self, sacc_path: str) -> dict:
        """
        Build the tjpcov config dict from self._cfg.
        Single source of truth — no parameter duplication.
        """
        cosmo  = self._cosmo
        mor    = self._cfg["mass_observable"]
        modeling = self._cfg["counts_modeling"]
        survey = self._survey

        return {
            "tjpcov": {
                "use_mpi": False,
                "do_xi": False,
                "cov_type": ["ClusterCountsGaussian", "ClusterCountsSSC"],
                "sacc_file": sacc_path,
                "cosmo": "set",
                "outdir": survey.get("outdir", "./"),
            },
            "parameters": {
                "Omega_c":            cosmo["Omega_m"] - cosmo["Omega_b"],
                "Omega_b":            cosmo["Omega_b"],
                "h":                  cosmo["h"],
                "n_s":                cosmo["n_s"],
                "sigma8":             cosmo["sigma8"],
                "w0":                 cosmo.get("w0", -1.0),
                "wa":                 cosmo.get("wa", 0.0),
                "transfer_function":  cosmo.get("transfer_function", "boltzmann_camb"),
            },
            "mor_parameters" :{
                "mass_func": modeling["mass_func"],
                "mass_def": modeling["mass_def"],
                "halo_bias": modeling["halo_bias"],
                "min_halo_mass": 10**float(modeling["mass_min"]),
                "max_halo_mass": 10**float(modeling["mass_max"]),
                "m_pivot": mor["pivot_mass"],
                "z_pivot": mor["pivot_redshift"],
                "mu_p0": mor["mu0"],
                "mu_p1": mor["mu1"],
                "mu_p2": mor["mu2"],
                "sigma_p0": mor["sigma0"],
                "sigma_p1": mor["sigma1"],
                "sigma_p2": mor["sigma2"],
            },
            "photo-z":{
            "sigma_0": modeling["photoz_err"]
            },
        }

    def setup(self) -> "ClusterCountsForecast":
        """Build theta0, bins, model, covariance, ForecastKit. Called automatically by run()."""
        self.theta0 = self._build_theta0()
        self._model_fn, self.z_bins, self.proxy_bins = self._build_model_fn()
        self._counts_fid = self._model_fn(self.theta0)
        self._cov = self._build_covariance()
        self._sigma_map = self._resolve_sigma_map()
        self._fk = ForecastKit(
            function=self._model_fn, theta0=self.theta0, cov=self._cov
        )
        self._is_setup = True
        return self

    # ------------------------------------------------------------------
    # Priors
    # ------------------------------------------------------------------

    def set_priors(self, sigma_map: Optional[dict] = None) -> "ClusterCountsForecast":
        """
        Override priors at runtime (replaces the yaml priors section).
        Returns self for chaining.
        """
        if sigma_map is not None:
            self._priors = {"sigma_map": sigma_map}
        return self

    def _resolve_sigma_map(self) -> dict:
        """
        Build the final {param: sigma} dict.
        Priority: sigma_map entries in yaml > prior_frac fallback.
        """
        sigma_map = {}
        prior_frac = self._priors.get("prior_frac")
        yaml_sigma_map = self._priors.get("sigma_map", {}) or {}

        if prior_frac is not None:
            if prior_frac <= 0:
                raise ValueError("prior_frac must be positive.")
            for p in self.active_params:
                ref = abs(self._fiducial_value(p))
                if ref == 0.0:
                    raise ValueError(
                        f"Cannot build relative prior for '{p}': fiducial value is zero."
                    )
                sigma_map[p] = prior_frac * ref

        for p, sigma in yaml_sigma_map.items():
            if sigma is None:
                continue
            if p not in self.active_params:
                raise ValueError(
                    f"Prior defined for '{p}' but it is not in active_params."
                )
            if sigma <= 0:
                raise ValueError(f"Prior sigma for '{p}' must be positive.")
            sigma_map[p] = float(sigma)

        return sigma_map

    def _build_fisher_prior_matrix(self) -> np.ndarray:
        n = len(self.active_params)
        fisher_prior = np.zeros((n, n), dtype=float)
        for i, p in enumerate(self.active_params):
            if p in self._sigma_map:
                fisher_prior[i, i] = 1.0 / self._sigma_map[p] ** 2
        return fisher_prior

    def _build_dali_prior_terms(self) -> tuple:
        """Returns (prior_terms, prior_bounds) or (None, None)."""
        if not self._sigma_map:
            return None, None

        n = len(self.active_params)
        mean = np.array(self.theta0, dtype=float)
        cov_prior = np.zeros((n, n), dtype=float)

        for i, p in enumerate(self.active_params):
            cov_prior[i, i] = self._sigma_map[p] ** 2 if p in self._sigma_map else 1e30

        prior_terms = [("gaussian", {"mean": mean, "cov": cov_prior})]

        nsigma = self._priors.get("nsigma_bounds")
        prior_bounds = None
        if nsigma is not None:
            prior_bounds = []
            for i, p in enumerate(self.active_params):
                if p in self._sigma_map:
                    s = self._sigma_map[p]
                    prior_bounds.append(
                        (float(self.theta0[i] - nsigma * s),
                         float(self.theta0[i] + nsigma * s))
                    )
                else:
                    prior_bounds.append((-np.inf, np.inf))

        return prior_terms, prior_bounds

    # ------------------------------------------------------------------
    # Core forecast
    # ------------------------------------------------------------------

    def run(self, mode: Optional[str] = None) -> "ClusterCountsForecast":
        """
        Compute Fisher and/or DALI forecast.
        mode overrides forecast.mode from yaml if provided.
        Returns self for chaining.
        """
        if not self._is_setup:
            self.setup()

        mode = mode or self._fcast.get("mode", "both")
        names  = self.active_params
        labels = [PARAM_LABELS[p] for p in self.active_params]

        if mode in ("fisher", "both"):
            self._run_fisher(names, labels)

        if mode in ("dali", "both"):
            self._run_dali(names, labels)

        return self

    def _run_fisher(self, names: list, labels: list) -> None:
        d = self._deriv
        fisher = self._fk.fisher(
            method=d["method"],
            stepsize=d["stepsize"],
            num_points=d["num_points"],
            extrapolation=d["extrapolation"],
            levels=d["levels"],
        )
        self.fisher_matrix = fisher
        fisher_prior = self._build_fisher_prior_matrix()
        self.fisher_posterior = fisher + fisher_prior

        label = "Fisher" if not self._sigma_map else "Fisher + prior"
        gnd = self._fk.getdist_fisher_gaussian(
            fisher=self.fisher_posterior,
            names=names,
            labels=labels,
            label=label,
        )
        self._getdist_objects.append(gnd)
        self._legend_labels.append(label)
        self._contour_colors.append("#f21901")

    def _run_dali(self, names: list, labels: list) -> None:
        d = self._deriv
        dali = self._fk.dali(
            forecast_order=d["forecast_order"],
            method=d["method"],
            stepsize=d["stepsize"],
            num_points=d["num_points"],
            extrapolation=d["extrapolation"],
            levels=d["levels"],
        )
        self.dali_result = dali

        prior_terms, prior_bounds = self._build_dali_prior_terms()
        label = "DALI" if not self._sigma_map else "DALI + prior"

        kwargs = dict(dali=dali, names=names, labels=labels, label=label)
        if prior_terms is not None:
            kwargs["prior_terms"] = prior_terms
        if prior_bounds is not None:
            kwargs["prior_bounds"] = prior_bounds

        samples = self._fk.getdist_dali_emcee(**kwargs)
        self._getdist_objects.append(samples)
        self._legend_labels.append(label)
        self._contour_colors.append("#e1af00")

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        print(f"\nactive_params : {self.active_params}")
        print(f"theta0        : {self.theta0}")
        print(f"z_bins        : {self.z_bins}")
        print(f"proxy_bins    : {self.proxy_bins}")
        print(f"counts (fid)  : {self._counts_fid}")

        if not self._sigma_map:
            print("No Gaussian priors applied.")
        else:
            print("Gaussian priors:")
            for p in self.active_params:
                if p in self._sigma_map:
                    print(f"  {p:>15s} : sigma_prior = {self._sigma_map[p]:.4g}")

        if self.fisher_posterior is not None:
            cov_params = np.linalg.inv(self.fisher_posterior)
            sigma_params = np.sqrt(np.diag(cov_params))
            corr = cov_params / np.outer(sigma_params, sigma_params)
            print("\nFisher marginalized sigmas:")
            for p, s in zip(self.active_params, sigma_params):
                print(f"  {p:>15s} : {s:.4g}")
            print("\nFisher correlation matrix:")
            print(np.round(corr, 3))

    def save_yaml(self, path: Union[str, Path]) -> None:
        """Write back the config that produced this forecast."""
        with open(path, "w") as f:
            yaml.dump(self._cfg, f, default_flow_style=False)

    # ------------------------------------------------------------------
    # Plotting — single forecast
    # ------------------------------------------------------------------

    def plot(
        self,
        output: Optional[str] = None,
        filled: Optional[bool] = None,
        show: Optional[bool] = None,
    ) -> None:
        """Triangle plot for this single forecast."""
        if not self._getdist_objects:
            raise RuntimeError("No results to plot. Call run() first.")

        output = output or self._plot.get("output", "triangle.pdf")
        filled = filled if filled is not None else self._plot.get("filled", False)
        show   = show   if show   is not None else self._plot.get("show", True)

        names = self.active_params
        width = 3.8 if len(names) <= 2 else 7.0

        plotter = getdist_plots.get_subplot_plotter(width_inch=width)
        plotter.settings.linewidth_contour = 1.2
        plotter.settings.linewidth = 1.2
        plotter.settings.figure_legend_frame = False
        plotter.settings.legend_rect_border = False

        plotter.triangle_plot(
            self._getdist_objects,
            params=names,
            filled=filled,
            contour_colors=self._contour_colors,
            legend_labels=self._legend_labels,
        )

        plotter.export(output)
        print(f"Saved plot: {output}")

        if show:
            plt.show()

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------

    @classmethod
    def run_sweep(
        cls,
        yaml_path: Union[str, Path],
        sweep: Optional[dict] = None,
        mode: Optional[str] = None,
    ) -> list:
        """
        Run a Cartesian sweep over any combination of parameters and
        overlay all resulting contours in one triangle plot.

        sweep overrides the yaml 'sweep' section if provided.
        Returns a list of ClusterCountsForecast instances (one per case).

        Example:
            forecasts = ClusterCountsForecast.run_sweep(
                "config.yaml",
                sweep={"sky_area": [440, 1000, 5000]},
            )
        """
        base_cfg = cls._load_yaml(yaml_path)
        sweep = sweep or base_cfg.get("sweep")

        if not sweep:
            raise ValueError("No sweep defined in yaml or passed as argument.")

        cases = cls._cartesian_cases(sweep)
        print(f"Sweep: {len(cases)} case(s) — {list(sweep.keys())}")

        plot_cfg = base_cfg.get("plot", {})
        colormap  = plot_cfg.get("colormap", "cmr.pride")
        cmap_min  = plot_cfg.get("cmap_min", 0.15)
        cmap_max  = plot_cfg.get("cmap_max", 0.85)
        filled    = plot_cfg.get("filled", False)
        output    = plot_cfg.get("output", "sweep_overlay.pdf")
        show      = plot_cfg.get("show", True)

        colors = cmr.take_cmap_colors(
            colormap,
            len(cases),
            cmap_range=(cmap_min, cmap_max),
            return_fmt="hex",
        )

        forecasts = []
        all_getdist = []
        all_colors  = []
        all_labels  = []

        active_params = list(base_cfg["forecast"]["active_params"])
        names  = active_params
        labels = [PARAM_LABELS[p] for p in active_params]

        for case, color in zip(cases, colors):
            label = cls._format_case_label(case)
            print(f"\n{'='*60}\nCase: {label}\n{'='*60}")

            overrides = cls._case_to_overrides(case)
            cfg = cls._deep_merge(base_cfg, overrides)

            # write a temporary yaml so the constructor can load it
            import tempfile, json
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as tmp:
                yaml.dump(cfg, tmp)
                tmp_path = tmp.name

            forecast = cls(tmp_path)
            if mode is not None:
                cfg["forecast"]["mode"] = mode
            forecast.run()
            forecast.print_summary()

            # collect getdist objects and re-label them with the case label
            for gd_obj, base_label in zip(
                forecast._getdist_objects, forecast._legend_labels
            ):
                gd_obj.label = f"{base_label}: {label}"
                all_getdist.append(gd_obj)
                all_colors.append(color)
                all_labels.append(f"{base_label}: {label}")

            forecasts.append(forecast)
            os.unlink(tmp_path)

        # overlay plot
        width = 3.8 if len(names) <= 2 else 7.0
        plotter = getdist_plots.get_subplot_plotter(width_inch=width)
        plotter.settings.linewidth_contour = 2.5
        plotter.settings.linewidth = 2.5
        plotter.settings.figure_legend_frame = False
        plotter.settings.legend_rect_border = False
        plotter.settings.legend_fontsize = 16

        plotter.triangle_plot(
            all_getdist,
            params=names,
            filled=filled,
            contour_colors=all_colors,
            legend_labels=all_labels,
        )

        plotter.export(output)
        print(f"\nSaved sweep plot: {output}")

        if show:
            plt.show()

        return forecasts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Cluster-count forecast from a yaml config file."
    )
    parser.add_argument("config", help="Path to yaml config file.")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run the sweep defined in the yaml (or pass --sweep-params).",
    )
    parser.add_argument(
        "--mode",
        choices=["fisher", "dali", "both"],
        default=None,
        help="Override forecast.mode from yaml.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override plot.output from yaml.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting.",
    )
    args = parser.parse_args()

    if args.sweep:
        forecasts = ClusterCountsForecast.run_sweep(args.config, mode=args.mode)
    else:
        forecast = ClusterCountsForecast(args.config)
        forecast.run(mode=args.mode)
        forecast.print_summary()
        if not args.no_plot:
            forecast.plot(output=args.output)


if __name__ == "__main__":
    main()
