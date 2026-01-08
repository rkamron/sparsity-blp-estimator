"""
Run Monte Carlo simulations for multiple estimators (BLP + IV, BLP − IV, Shrinkage).
Simulates a dataset under a chosen DGP,
runs the BLP estimator with a chosen IV,
runs the Shrinkage estimator,
aggregates estimates into mean, bias, and stdev.
"""

import numpy as np

from simulation.config import SimConfig
from simulation.simulate import simulate_dataset
from estimators.blp import estimate_blp_sigma
from estimators.shrinkage import estimate_shrinkage_sigma

def init_storage(R, include_shrinkage=False):
    """
    Pre-allocate arrays to store parameter estimates across all monte carlo loops.
    
    :param R: Number of replications
    :param include_shrinkage: Whether to include storage for shrinkage estimator results
    """
    storage = {
        "sigma": np.zeros(R),
        "beta_p": np.zeros(R),
        "beta_w": np.zeros(R),
    }
    if include_shrinkage:
        storage["sigma_shrink"] = np.zeros(R)
        storage["beta_p_shrink"] = np.zeros(R)
        storage["beta_w_shrink"] = np.zeros(R)
    return storage

def run_mc_cell(
    DGP,
    T,
    J,
    R_mc=50,
    iv_type="cost",
    seed=123,
    run_shrinkage=False
):
    """
    One Monte Carlo cell for BLP estimation.
    
    :param DGP: string like 'DGP1' from 1-4.
    :param T: Number of markets
    :param J: Number of products per market
    :param R_mc: Number of Monte Carlo runs
    :param iv_type: BLP estimation type, with or without IV
    :param seed: 
    :param run_shrinkage: Whether to run shrinkage estimator
    """
    np.random.seed(seed)
    cfg = SimConfig()

    # Prepare arrays for σ̂, β̂_p, β̂_w (and shrinkage if requested)
    results = init_storage(R_mc, include_shrinkage=run_shrinkage)

    # Monte Carlo loop
    # 1. simulate dataset
    # 2. estimate BLP
    # 3. store results
    for r in range(R_mc):
        # simulate new markets dataset
        markets = simulate_dataset(DGP, T=T, J=J, cfg=cfg)

        # estimate BLP parameters
        sigma_hat, beta_hat, _ = estimate_blp_sigma(
            markets,
            iv_type=iv_type,
            R=cfg.R0
        )

        # store results
        results["sigma"][r]  = sigma_hat
        results["beta_p"][r] = beta_hat[1]  # price coefficient
        results["beta_w"][r] = beta_hat[2]  # characteristic coefficient

        if run_shrinkage:
            sigma_s, beta_s, _, _ = estimate_shrinkage_sigma(
                markets,
                R=cfg.R0,
                n_iter=200,
                burn=100
            )
            results["sigma_shrink"][r]  = sigma_s
            results["beta_p_shrink"][r] = beta_s[1]
            results["beta_w_shrink"][r] = beta_s[2]

        print(f"[{DGP} | {iv_type}] replication {r+1}/{R_mc}")

    return results

def summarize_mc(results, cfg):
    """
    Convert raw monte carlo results into table-1 style.
    Return dict summary with mean, bias, sd for each estimated parameter.
    mean: average estimate across loops
    bias: mean - true value
    sd: monte carlo dispersion
    
    :param results: Raw monte carlo results
    :param cfg: Configuration object
    """
    summary = {}
    for k, est in results.items():
        true_name = k.replace("_shrink", "")
        summary[k] = {
            "mean": np.mean(est),
            "bias": np.mean(est) - getattr(cfg, f"{true_name}_star"),
            "sd": np.std(est, ddof=1),
        }
    return summary

def run_table1_cell():
    """
    Running a row of Table 1 with the configs below, including BLP + IV, BLP - IV, and Shrinkage.

    Adjust DGP, T, J as needed.
    """
    cfg = SimConfig()

    DGP = "DGP1"    # options: DGP1, DGP2, DGP3, DGP4
    T = 25          # number of markets
    J = 15          # number of products per market

    print("Running BLP with cost IV")
    res_cost = run_mc_cell(
        DGP=DGP,
        T=T,
        J=J,
        iv_type="cost"
    )

    print("Running BLP without cost IV")
    res_nocost = run_mc_cell(
        DGP=DGP,
        T=T,
        J=J,
        iv_type="nocost"
    )

    print("Running Shrinkage (no IV)")
    res_shrink = run_mc_cell(
        DGP=DGP,
        T=T,
        J=J,
        iv_type="nocost",
        run_shrinkage=True
    )

    sum_cost   = summarize_mc(res_cost, cfg)
    sum_nocost = summarize_mc(res_nocost, cfg)
    sum_shrink = summarize_mc(res_shrink, cfg)

    print("\n=== TABLE 1 CELL ===")
    print("BLP + cost IV:", sum_cost)
    print("BLP - cost IV:", sum_nocost)
    print("Shrinkage:", sum_shrink)

if __name__ == "__main__":
    run_table1_cell()
