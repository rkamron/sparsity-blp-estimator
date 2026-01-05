"""
Run Monte Carlo simulations for BLP estimation.
Simulates a dataset under a chose DGP, 
run the BLP estimator with a chosen IV, 
aggreagate estimates into mean, bias, and stdev.
"""

import numpy as np

from simulation.config import SimConfig
from simulation.simulate import simulate_dataset
from blp import estimate_blp_sigma

def init_storage(R):
    """
    Pre-allocate arrays to store parameter estimates across all monte carlo loops.
    
    :param R: Number of replications
    """
    return {
        "sigma": np.zeros(R),
        "beta_p": np.zeros(R),
        "beta_w": np.zeros(R),
    }

def run_mc_cell(
    DGP,
    T,
    J,
    R_mc=50,
    iv_type="cost",
    seed=123
):
    """
    One Monte Carlo cell for BLP estimation.
    
    :param DGP: string like 'DGP1' from 1-4.
    :param T: Number of markets
    :param J: Number of products per market
    :param R_mc: Number of Monte Carlo runs
    :param iv_type: BLP estimation type, with or without IV
    :param seed: 
    """
    np.random.seed(seed)
    cfg = SimConfig()

    # Prepare arrays for σ̂, β̂_p, β̂_w
    results = init_storage(R_mc)

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


        print(f"[{DGP} | {iv_type}] replication {r+1}/{R_mc}")

    return results

def summarize_mc(results, cfg):
    """
    Convert raw monte carlo results into table-1 style.
    Return dict summary with mean, bias, sd for each estimated parameter.
    mean: average estimate across loops
    bias: mean - true value
    sd: monte carlo dispersion
    {
        "sigma": {"mean": ..., "bias": ..., "sd": ...},
        "beta_p": {"mean": ..., "bias": ..., "sd": ...},
        "beta_w": {"mean": ..., "bias": ..., "sd": ...},
    }
    
    :param results: Raw monte carlo results
    :param cfg: Configuration object
    """
    summary = {"sigma": {}, "beta_p": {}, "beta_w": {}}
    for k in summary.keys():
        est = results[k]
        summary[k] = {
            "mean": np.mean(est),
            "bias": np.mean(est) - getattr(cfg, f"{k}_star"),
            "sd": np.std(est, ddof=1),
        }
    return summary

def run_table1_cell():
    """
    Running a row of Table 1 with the configs below.
    """
    cfg = SimConfig()

    DGP = "DGP1"
    T = 25
    J = 15

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

    sum_cost   = summarize_mc(res_cost, cfg)
    sum_nocost = summarize_mc(res_nocost, cfg)

    print("\n=== TABLE 1 CELL ===")
    print("BLP + cost IV:", sum_cost)
    print("BLP - cost IV:", sum_nocost)

if __name__ == "__main__":
    run_table1_cell()

