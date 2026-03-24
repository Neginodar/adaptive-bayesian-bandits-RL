# import pymc as pm
# import numpy as np
# import arviz as az
# import matplotlib.pyplot as plt


# def build_bandit_pgm(arm_ids: np.ndarray, rewards: np.ndarray, n_arms: int):
#     """
#     Bayesian Network (PGM) over bandit reward structure.
    
#     Graph structure:
#         mu_global --> theta_a --> r_t
#         sigma_global --> theta_a
    
#     This is a proper hierarchical PGM with:
#     - Hyperpriors (mu_global, sigma_global)
#     - Latent variables (theta_a per arm)  
#     - Observed data (rewards)
#     """
#     with pm.Model() as pgm:
#         # Hyperprior nodes
#         mu_global    = pm.Beta("mu_global", alpha=2, beta=2)
#         kappa_global = pm.HalfNormal("kappa_global", sigma=10)

#         # Latent arm nodes — drawn from shared hyperprior (partial pooling)
#         alpha_arms = pm.Deterministic("alpha_arms", mu_global * kappa_global)
#         beta_arms  = pm.Deterministic("beta_arms",  (1 - mu_global) * kappa_global)

#         theta = pm.Beta("theta", alpha=alpha_arms, beta=beta_arms, shape=n_arms)

#         # Observed reward node
#         pm.Bernoulli("obs", p=theta[arm_ids], observed=rewards)

#         trace = pm.sample(
#             1000, tune=1000, chains=2,
#             target_accept=0.9, progressbar=True,
#             return_inferencedata=True
#         )

#     return pgm, trace


# def plot_pgm_posteriors(trace, n_arms: int, true_probs: list):
#     """Plot posterior distributions over arm reward probabilities."""
#     fig, axes = plt.subplots(1, n_arms, figsize=(4 * n_arms, 4))
#     for a in range(n_arms):
#         samples = trace.posterior["theta"].values[:, :, a].flatten()
#         axes[a].hist(samples, bins=40, color="royalblue", alpha=0.7, density=True)
#         axes[a].axvline(true_probs[a], color="red", linestyle="--",
#                         label=f"True p={true_probs[a]}")
#         axes[a].set_title(f"Arm {a} posterior")
#         axes[a].set_xlabel("θ"); axes[a].legend(fontsize=8)
#     plt.suptitle("PGM Posterior: P(θ_a | data)", fontsize=13)
#     plt.tight_layout()
#     plt.savefig("results/pgm_posteriors.png", dpi=150)
#     print("PGM plot saved to results/pgm_posteriors.png")
#     az.plot_posterior(trace, var_names=["theta", "mu_global"])
#     plt.savefig("results/pgm_arviz.png", dpi=150)
#     print("ArviZ posterior saved to results/pgm_arviz.png")
import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import os


def build_bandit_pgm(arm_ids: np.ndarray, rewards: np.ndarray, n_arms: int):
    """
    Hierarchical Bayesian Network (PGM) over bandit reward structure.

    Graph structure:
        mu_global ──┐
                    ├──► theta_a ──► r_t  (observed)
        kappa ──────┘

    Hyperpriors:  mu_global, kappa_global
    Latent nodes: theta_a per arm (Beta distributed)
    Observed:     r_t (Bernoulli rewards)
    """
    with pm.Model() as pgm:
        mu_global    = pm.Beta("mu_global", alpha=2, beta=2)
        kappa_global = pm.HalfNormal("kappa_global", sigma=10)

        alpha_arms = pm.Deterministic("alpha_arms", mu_global * kappa_global)
        beta_arms  = pm.Deterministic("beta_arms",  (1 - mu_global) * kappa_global)

        theta = pm.Beta("theta", alpha=alpha_arms, beta=beta_arms, shape=n_arms)
        pm.Bernoulli("obs", p=theta[arm_ids], observed=rewards)

        trace = pm.sample(1000, tune=1000, chains=4, cores=4,
                  target_accept=0.9, return_inferencedata=True)
    return pgm, trace


def plot_pgm_posteriors(trace, n_arms: int, true_probs: list):
    """Plot posterior distributions over arm reward probabilities."""
    os.makedirs("results", exist_ok=True)

    fig, axes = plt.subplots(1, n_arms, figsize=(4 * n_arms, 4))
    for a in range(n_arms):
        samples = trace.posterior["theta"].values[:, :, a].flatten()
        axes[a].hist(samples, bins=40, color="royalblue", alpha=0.7, density=True)
        axes[a].axvline(true_probs[a], color="red", linestyle="--",
                        label=f"True p={true_probs[a]}")
        axes[a].set_title(f"Arm {a} posterior")
        axes[a].set_xlabel("θ")
        axes[a].legend(fontsize=8)

    plt.suptitle("PGM Posterior: P(θ_a | data) vs True Probabilities", fontsize=13)
    plt.tight_layout()
    plt.savefig("results/pgm_posteriors.png", dpi=150, bbox_inches="tight")
    print("PGM posteriors saved to results/pgm_posteriors.png")

    az.plot_posterior(trace, var_names=["theta", "mu_global"])
    plt.savefig("results/pgm_arviz.png", dpi=150, bbox_inches="tight")
    print("ArviZ posterior saved to results/pgm_arviz.png")
    plt.close("all")
