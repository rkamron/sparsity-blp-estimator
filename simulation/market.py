# simulation/market.py
# Structural DGP

import numpy as np
from .dgp import generate_eta_alpha


def simulate_market(dgp, J, cfg):
    """
    Simulate a single market.
    Generates product characteristics, prices, demand shocks, and cost shocks.

    :param dgp: string like 'DGP1' from 1-4.
    :param J: Number of products
    :param cfg: Configuration object
    """
    # Exogenous product characteristic for every project J in this market
    w = np.random.uniform(cfg.w_low, cfg.w_high, size=J)

    # Cost shock 
    u = np.random.normal(0.0, cfg.cost_sd, size=J)

    # Demand shocks and endogeneity
    eta_star, alpha_star = generate_eta_alpha(dgp, J, cfg)

    # Price equation
    p = alpha_star + 0.3 * w + u

    # Demand shock
    xi = cfg.xi_bar_star + eta_star
    
    return {
        "w": w.astype(float),
        "p": p.astype(float),
        "u": u.astype(float),
        "xi": xi.astype(float), 
        "eta": eta_star.astype(float),
        "alpha": alpha_star.astype(float)
    }