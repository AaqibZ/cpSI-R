# cpSI-R Reproducibility Package (Minimal)
This repository provides a minimal, well-documented Python implementation for reproducing experiments from the cpSI-R paper.
It is intentionally compact and modular so reviewers and readers can run, inspect, and extend it.

## Contents
- `run_experiment.py` : high-level experiment runner (config-driven)
- `cpsir.py` : cpSI-R diffusion model and simulator
- `sampling.py` : temporal sampling utilities (Jaccard + Kulczynski)
- `calc_influence.py` : candidate influence estimation (greedy)
- `baselines.py` : simple baseline implementations (degree, dynamic-degree-discount placeholder)
- `data_prep.py` : minimal dataset loader and normalizer (expected `data/` folder)
- `plot_results.py` : plotting utilities to reproduce figures
- `example_config.json` : example experiment configuration

## How to run (example)
1. Place your temporal dataset(s) in `data/` as csv files with header: `src,tgt,timestamp`
2. Edit `example_config.json` to point to your dataset and parameters.
3. Run:
```
python run_experiment.py --config example_config.json
```

## Notes
- This package is a **minimal** reproducibility scaffold. The actual research code used for large-scale experiments may include optimized C/C++ modules and cluster orchestration.
- The cpSI-R model here is a transparent Python implementation matching the paper's conceptual description (reinforcement, time-limited activity, reactivation).
- Please see Appendix C in the paper for parameter choices and experiment details.

