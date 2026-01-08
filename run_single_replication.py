# run_single_replication.py
# one Monte Carlo draw
# simulate data, estimate BLP and shrinkage, print results

import numpy as np
from simulation.config import SimConfig
from simulation.simulate import simulate_dataset
from estimators.blp import estimate_blp_sigma
from estimators.shrinkage import estimate_shrinkage_sigma


def main():
    np.random.seed(123) 

    cfg = SimConfig()

    # simulation parameters (paper grid) 
    DGP = "DGP1"   # sparse ξ, exogenous price
    T = 25
    J = 15

    print(f"Running {DGP} with T={T}, J={J}")

    # simulate data
    markets = simulate_dataset(DGP, T=T, J=J, cfg=cfg)

    s0 = 1 - markets[0]["s"].sum()
    corr_px = np.corrcoef(markets[0]["p"], markets[0]["xi"])[0, 1]

    print("Market 0:")
    print(markets[0])
    print("  sum shares =", markets[0]["s"].sum())
    print("  outside share =", s0)
    print("  corr(p, xi) =", corr_px)

    # BLP estimation
    sigma_hat, beta_hat, obj = estimate_blp_sigma(
        markets,
        iv_type="cost",  # THIS IS THE IMPORTANT ONE
        R=cfg.R0
    )

    print("\nBLP (with cost IV) results")
    print("  sigma_hat =", sigma_hat)
    print("  beta_hat =", beta_hat)
    print("  GMM obj   =", obj)

    print("\nTrue parameters")
    print("  sigma* =", cfg.sigma_star)
    print("  beta_p* =", cfg.beta_p_star)
    print("  beta_w* =", cfg.beta_w_star)

    # Shrinkage estimation (no IV, sparse xi)
    sigma_shrink, beta_shrink, score_shrink, gamma_prob = estimate_shrinkage_sigma(
        markets,
        R=cfg.R0,
        n_iter=200,
        burn=100,
        v0=1e-4,
        v1=1.0
    )

    print("\nShrinkage results (sparse xi)")
    print("  sigma_hat =", sigma_shrink)
    print("  beta_hat =", beta_shrink)
    print("  avg inclusion prob =", gamma_prob.mean())
    print("  shrinkage score =", score_shrink)


if __name__ == "__main__":
    main()