# CrowKit

CrowKit is a manager for building Fisher and beyond-Fisher forecasts using CROW, TJPCov, and DerivKit.

It connects observable modelling, covariance computation, and derivative-based forecasting into a single, 
reproducible workflow.

CrowKit does not implement the underlying physics. It orchestrates existing tools.

---

## Dependencies

CrowKit relies on:

- [CROW](https://github.com/LSSTDESC/crow) – observable modelling  
- [TJPCov](https://github.com/LSSTDESC/tjpcov) – covariance calculation  
- [DerivKit](https://github.com/derivkit/derivkit) – numerical derivatives and forecasting (including beyond-Fisher methods such as DALI)  

---

## Installation

```bash
pip install -e .
```


## Quickstart
```bash
crowkit-cluster-counts-forecast examples/cluster_counts_forecast/cluster_config.yaml --mode both
```
The mode option can be both, fisher or dali. You can also run it as a class inside a python file.
```bash
python examples/cluster_counts_forecast/run_forecast.py
```
## Scope
CrowKit is designed to:
- manage forecast configurations
- handle parameter bookkeeping
- build data vectors and covariances via external tools
- compute derivatives
- assemble Fisher matrices
It is a manager layer, _not_ a replacement for CROW, TJPCov, or DerivKit.
