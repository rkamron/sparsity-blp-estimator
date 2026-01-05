import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .market import simulate_market

tfd = tfp.distributions


def simulate_shares(p, w, xi, cfg):
    """
    Simulate market shares for consumer behavior using 
    random coefficients logit model, 
    u_ijt = β_pi * p_jt + β_w * w_jt + ξ_jt + eta_ijt,
    and computes s from equation 11

    :param p: Prices
    :param w: Characteristics
    :param xi: Demand shocks
    :param cfg: Configuration object
    """
    p  = tf.convert_to_tensor(p,  dtype=tf.float64)
    w  = tf.convert_to_tensor(w,  dtype=tf.float64)
    xi = tf.convert_to_tensor(xi, dtype=tf.float64)

    #β_pi ∼ N(β_p_*, σ^∗2) 
    beta_p_draws = tf.cast(
        tfd.Normal(cfg.beta_p_star, cfg.sigma_star).sample(cfg.R0),
        tf.float64
    )

    # u_ijt for all R0 draws
    # shape (R0, J)
    # each row is one consumer and each column is one product
    # u_ijt = β_pi * p_jt + β_w * w_jt + ξ_jt
    util = (
        tf.expand_dims(beta_p_draws, 1) * tf.expand_dims(p, 0)
        + cfg.beta_w_star * tf.expand_dims(w, 0)
        + tf.expand_dims(xi, 0)
    )

    # choice probabilities
    # P_ijt = exp(u_ijt) / (1 + sum_k exp(u_ikt))
    expu = tf.exp(util)
    denom = 1.0 + tf.reduce_sum(expu, axis=1, keepdims=True)
    probs = expu / denom

    # aggreagate to market shares
    # s_jt = 1/R0 sum_i_to_R0 P_ijt
    shares = tf.reduce_mean(probs, axis=0)

    return shares.numpy()


def finalize_market(market, cfg):
    """
    Convert structural primatives into observable market data.
    
    :param market: Market structural primatives
    :param cfg: Configuration object
    """
    s = simulate_shares(market["p"], market["w"], market["xi"], cfg)

    # convert shares to quantities
    # q_jt = N_t * s_jt
    q = np.round(cfg.Nt * s).astype(int)

    return {
        "s": s,                     # market shares
        "p": market["p"],           # prices  
        "w": market["w"],           # characteristics
        "u": market["u"],           # marginal costs
        "q": q,                     # quantities sold
        "xi": market["xi"],
        "eta": market["eta"],
        "alpha": market["alpha"],
    }


def simulate_dataset(dgp, T, J, cfg):
    """
    Simulate a dataset with T markets of J products each.

    :param dgp: string like 'DGP1' from 1-4.
    :param T: Number of markets
    :param J: Number of products
    :param cfg: Configuration object
    """
    markets = []

    # simulate T markets
    for _ in range(T):
        base = simulate_market(dgp, J, cfg)
        markets.append(finalize_market(base, cfg))

    print(f"Simulated dataset with {T} markets of {J} products each under DGP {dgp}.")
    return markets