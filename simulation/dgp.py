"""DGP functions for generating eta and alpha."""

import numpy as np

def generate_eta_alpha(dgp, J, cfg):
    """
    Generate eta_star and alpha_star based on the DGP type.
    Tells us how demand shocks are distrubuted and how endogenous prices are.
    Returns:
        eta_star: np.ndarray shape (J,)
        alpha_star: np.ndarray shape (J,)
    """
    #both initialized to zeros
    eta = np.zeros(J)
    alpha = np.zeros(J)

    # As specified in paper Section 4.1
    # DGP1 and DGP2 both get sparse eta.
    # eta has first 40% non-zero values alternating +1, -1.
    # DGP2 has endogenous prices: alpha = scale * sign(eta)
    if dgp in ["DGP1", "DGP2"]:
        cutoff = int(cfg.sparse_frac * J)

        for j in range(cutoff):
            eta[j] = cfg.eta_sparse_vals[j % 2]  # +1, -1 alternating

        if dgp == "DGP2":
            alpha = cfg.alpha_scale * np.sign(eta)

    elif dgp in ["DGP3", "DGP4"]:
        eta = np.random.normal(0.0, cfg.eta_dense_sd, size=J)

        if dgp == "DGP4":
            alpha = np.zeros(J)
            alpha[eta >= cfg.eta_dense_sd] = cfg.alpha_scale
            alpha[eta <= -cfg.eta_dense_sd] = -cfg.alpha_scale

    else:
        raise ValueError(f"Unknown DGP: {dgp}")

    return eta, alpha