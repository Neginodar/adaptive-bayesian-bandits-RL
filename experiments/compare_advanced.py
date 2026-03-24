import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pymc as pm
from scipy.stats import beta as beta_dist, norm as norm_dist
import arviz as az
import csv
from copy import deepcopy


# ── Config ────────────────────────────────────────────────────────────────────
N_ARMS      = 5
N_EPISODES  = 500
N_SEEDS     = 3
TRUE_MEANS  = [0.1, 0.3, 0.5, 0.7, 0.9]
TRUE_STDS   = [0.3, 0.3, 0.3, 0.3, 0.3]
CONTEXT_DIM = 4
WINDOW_SIZE = 50
MCMC_FREQ   = 50

initial_means = TRUE_MEANS.copy()

# ── Pre-generate shift schedule ───────────────────────────────────────────────
shift_schedule = {}
_rng = np.random.default_rng(seed=42)
for ep in range(1, N_EPISODES):
    if ep % 200 == 0:
        s = initial_means.copy()
        _rng.shuffle(s)
        shift_schedule[ep] = s
shift_episodes = list(shift_schedule.keys())


# ── Environments ──────────────────────────────────────────────────────────────
def pull_bernoulli(arm, probs):
    return int(np.random.random() < probs[arm])

def pull_gaussian(arm, means):
    return float(np.random.normal(means[arm], TRUE_STDS[arm]))

def pull_contextual(arm, context, weights):
    logit = context @ weights[arm]
    return int(np.random.random() < 1 / (1 + np.exp(-logit)))


# ── Agent 1: Beta-Bernoulli Thompson Sampling ─────────────────────────────────
class BernoulliBandit:
    def __init__(self, n_arms):
        self.n_arms  = n_arms
        self.alphas  = np.ones(n_arms)
        self.betas_p = np.ones(n_arms)
        self.history = {a: [] for a in range(n_arms)}
        self.traces  = {}
        self.posterior_snapshots = []

    def choose(self):
        return int(np.argmax(np.random.beta(self.alphas, self.betas_p)))

    def update(self, arm, reward):
        self.alphas[arm]  += reward
        self.betas_p[arm] += (1 - reward)
        self.history[arm].append(reward)

    def reset(self):
        self.alphas  = np.ones(self.n_arms)
        self.betas_p = np.ones(self.n_arms)

    def snapshot(self):
        self.posterior_snapshots.append(
            (self.alphas.copy(), self.betas_p.copy())
        )

    def run_mcmc(self, ep):
        for arm in range(self.n_arms):
            h = self.history[arm]
            if len(h) < 5: continue
            n, s = len(h), int(sum(h))
            with pm.Model():
                theta = pm.Beta("theta", alpha=1.0, beta=1.0)
                pm.Binomial("obs", n=n, p=theta, observed=s)
                # ✅ FIX: 4 chains
                trace = pm.sample(300, tune=300, chains=4, cores=4,
                                  progressbar=False, return_inferencedata=True)
            self.traces[(ep, arm)] = trace

    def posterior_entropy(self):
        return float(np.mean([beta_dist.entropy(a, b)
                               for a, b in zip(self.alphas, self.betas_p)]))


# ── Agent 2: Gaussian Hierarchical TS with sliding window ────────────────────
class GaussianHierarchicalBandit:
    def __init__(self, n_arms, window=WINDOW_SIZE):
        self.n_arms  = n_arms
        self.window  = window
        self.mu      = np.zeros(n_arms)
        self.sigma   = np.ones(n_arms) * 2.0
        self.history = {a: [] for a in range(n_arms)}
        self.traces  = {}

    def choose(self):
        return int(np.argmax(np.random.normal(self.mu, self.sigma)))

    def update(self, arm, reward):
        self.history[arm].append(reward)
        if len(self.history[arm]) > self.window:
            self.history[arm].pop(0)
        h = self.history[arm]
        n = len(h)
        self.mu[arm]    = np.mean(h)
        self.sigma[arm] = max(np.std(h) / np.sqrt(n) if n > 1 else 2.0, 0.01)

    def reset(self):
        self.mu    = np.zeros(self.n_arms)
        self.sigma = np.ones(self.n_arms) * 2.0

    def run_mcmc(self, ep):
        all_rewards, all_arms = [], []
        for a in range(self.n_arms):
            for r in self.history[a]:
                all_rewards.append(r)
                all_arms.append(a)
        if len(all_rewards) < 10: return
        rewards_arr = np.array(all_rewards)
        arms_arr    = np.array(all_arms, dtype=int)
        with pm.Model():
            mu_global    = pm.Normal("mu_global", mu=0, sigma=1)
            sigma_global = pm.HalfNormal("sigma_global", sigma=1)
            mu_arms      = pm.Normal("mu_arms", mu=mu_global,
                                     sigma=sigma_global, shape=self.n_arms)
            sigma_arms   = pm.HalfNormal("sigma_arms", sigma=1, shape=self.n_arms)
            pm.Normal("obs", mu=mu_arms[arms_arr],
                      sigma=sigma_arms[arms_arr], observed=rewards_arr)
            # ✅ FIX: 4 chains
            trace = pm.sample(300, tune=500, chains=4, cores=4,
                              progressbar=False, return_inferencedata=True,
                              target_accept=0.9)
        self.traces[ep] = trace

    def posterior_entropy(self):
        return float(np.mean([norm_dist.entropy(loc=m, scale=s)
                               for m, s in zip(self.mu, self.sigma)]))


# ── Agent 3: Contextual Bayesian TS with frequent MCMC ───────────────────────
class ContextualBayesianBandit:
    def __init__(self, n_arms, context_dim, mcmc_freq=MCMC_FREQ):
        self.n_arms      = n_arms
        self.context_dim = context_dim
        self.mcmc_freq   = mcmc_freq
        self.w_mu        = [np.zeros(context_dim) for _ in range(n_arms)]
        self.w_sigma     = [np.eye(context_dim)   for _ in range(n_arms)]
        self.history     = {a: [] for a in range(n_arms)}
        self.traces      = {}
        self.ep_count    = 0

    def choose(self, context):
        probs = []
        for a in range(self.n_arms):
            w = np.random.multivariate_normal(self.w_mu[a], self.w_sigma[a])
            probs.append(1 / (1 + np.exp(-(context @ w))))
        return int(np.argmax(probs))

    def update(self, arm, context, reward):
        self.history[arm].append((context, reward))
        self.ep_count += 1

    def reset(self):
        self.w_mu    = [np.zeros(self.context_dim) for _ in range(self.n_arms)]
        self.w_sigma = [np.eye(self.context_dim)   for _ in range(self.n_arms)]

    def maybe_run_mcmc(self, ep):
        if ep % self.mcmc_freq == 0:
            self.run_mcmc(ep)

    def run_mcmc(self, ep):
        for a in range(self.n_arms):
            data = self.history[a]
            if len(data) < 10: continue
            X = np.array([d[0] for d in data])
            y = np.array([d[1] for d in data], dtype=int)
            with pm.Model():
                w = pm.Normal("w", mu=0, sigma=1, shape=self.context_dim)
                p = pm.Deterministic("p", pm.math.sigmoid(X @ w))
                pm.Bernoulli("obs", p=p, observed=y)
                # ✅ FIX: 4 chains
                trace = pm.sample(300, tune=500, chains=4, cores=4,
                                  progressbar=False, return_inferencedata=True,
                                  target_accept=0.9)
            self.traces[(ep, a)] = trace
            self.w_mu[a]    = trace.posterior["w"].mean(dim=("chain","draw")).values
            self.w_sigma[a] = np.diag(
                trace.posterior["w"].std(dim=("chain","draw")).values ** 2
            )

    def posterior_entropy(self):
        entropies = []
        for a in range(self.n_arms):
            sign, logdet = np.linalg.slogdet(self.w_sigma[a])
            if sign > 0:
                k = self.context_dim
                entropies.append(0.5 * (k * (1 + np.log(2 * np.pi)) + logdet))
        return float(np.mean(entropies)) if entropies else 0.0


# ── Epsilon-Greedy Baseline ───────────────────────────────────────────────────
class EpsilonGreedyAgent:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms  = n_arms
        self.epsilon = epsilon
        self.counts  = np.zeros(n_arms)
        self.values  = np.zeros(n_arms)

    def choose(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        return int(np.argmax(self.values))

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

    def reset(self):
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)

    def posterior_entropy(self): return 0.0


# ── UCB Baseline ──────────────────────────────────────────────────────────────
class UCBAgent:
    """Sliding-window UCB — adapts to non-stationary reward distributions."""
    def __init__(self, n_arms, c=2.0, window=50):
        self.n_arms  = n_arms
        self.c       = c
        self.window  = window
        self.history = {a: [] for a in range(n_arms)}
        self.counts  = np.zeros(n_arms)
        self.values  = np.zeros(n_arms)

    def choose(self):
        # ✅ FIX: use window-local t, not global t
        total_window = max(1, sum(len(h) for h in self.history.values()))
        ucb = self.values + self.c * np.sqrt(
            np.log(total_window) / (self.counts + 1e-5)
        )
        return int(np.argmax(ucb))

    def update(self, arm, reward):
        self.history[arm].append(reward)
        if len(self.history[arm]) > self.window:
            self.history[arm].pop(0)
        h = self.history[arm]
        self.counts[arm] = len(h)
        self.values[arm] = np.mean(h)

    def reset(self):
        self.history = {a: [] for a in range(self.n_arms)}
        self.counts  = np.zeros(self.n_arms)
        self.values  = np.zeros(self.n_arms)

    def posterior_entropy(self): return 0.0


# ── Single-seed run ───────────────────────────────────────────────────────────
def run_single_seed(seed, run_mcmc_diag=False):
    np.random.seed(seed)
    TRUE_WEIGHTS = [np.random.randn(CONTEXT_DIM) for _ in range(N_ARMS)]

    agents = {
        "Beta-Bernoulli TS":        BernoulliBandit(N_ARMS),
        "Gaussian Hierarchical TS": GaussianHierarchicalBandit(N_ARMS),
        "Contextual Bayesian TS":   ContextualBayesianBandit(N_ARMS, CONTEXT_DIM),
        "Epsilon-Greedy":           EpsilonGreedyAgent(N_ARMS),
        "UCB":                      UCBAgent(N_ARMS, c=0.5),  # ← only this changes
        }

    env_types = {
        "Beta-Bernoulli TS":        "bernoulli",
        "Gaussian Hierarchical TS": "gaussian",
        "Contextual Bayesian TS":   "contextual",
        "Epsilon-Greedy":           "bernoulli",
        "UCB":                      "bernoulli",
    }

    results = {name: {"rewards": [], "regrets": [], "entropies": []}
               for name in agents}

    for name, agent in agents.items():
        current_means = initial_means.copy()
        env_type      = env_types[name]

        for ep in range(N_EPISODES):
            if ep in shift_schedule:
                current_means = shift_schedule[ep].copy()
                agent.reset()

            if env_type == "bernoulli":
                arm    = agent.choose()
                reward = pull_bernoulli(arm, current_means)
                agent.update(arm, reward)
            elif env_type == "gaussian":
                arm    = agent.choose()
                reward = pull_gaussian(arm, current_means)
                agent.update(arm, reward)
            elif env_type == "contextual":
                context = np.random.randn(CONTEXT_DIM)
                arm     = agent.choose(context)
                reward  = pull_contextual(arm, context, TRUE_WEIGHTS)
                agent.update(arm, context, reward)
                current_means = [
                    1/(1+np.exp(-(np.random.randn(CONTEXT_DIM) @ TRUE_WEIGHTS[a])))
                    for a in range(N_ARMS)
                ]

            results[name]["rewards"].append(float(reward))
            results[name]["regrets"].append(max(current_means) - current_means[arm])
            results[name]["entropies"].append(agent.posterior_entropy())

            if run_mcmc_diag and name == "Gaussian Hierarchical TS" \
                    and (ep + 1) % 100 == 0:
                agent.run_mcmc(ep + 1)

            if name == "Contextual Bayesian TS":
                agent.maybe_run_mcmc(ep)

            if run_mcmc_diag and name == "Beta-Bernoulli TS" and ep % 10 == 0:
                agent.snapshot()

        if run_mcmc_diag and name == "Beta-Bernoulli TS":
            agent.run_mcmc(N_EPISODES)

    return results, agents


# ── Multi-seed runner ─────────────────────────────────────────────────────────
print("Running multi-seed experiment...")
all_results = []
final_agents = None
for seed in range(N_SEEDS):
    print(f"  Seed {seed+1}/{N_SEEDS}...")
    is_last = (seed == N_SEEDS - 1)
    res, agents_obj = run_single_seed(seed, run_mcmc_diag=is_last)
    all_results.append(res)
    if is_last:
        final_agents = agents_obj


# ── Aggregate across seeds ────────────────────────────────────────────────────
agent_names = list(all_results[0].keys())
agg = {}
for name in agent_names:
    rewards_mat = np.array([r[name]["rewards"] for r in all_results])
    regrets_mat = np.array([r[name]["regrets"] for r in all_results])
    agg[name] = {
        "cum_reward_mean": np.cumsum(rewards_mat, axis=1).mean(axis=0),
        "cum_reward_std":  np.cumsum(rewards_mat, axis=1).std(axis=0),
        "cum_regret_mean": np.cumsum(regrets_mat, axis=1).mean(axis=0),
        "cum_regret_std":  np.cumsum(regrets_mat, axis=1).std(axis=0),
        "ep_regret_mean":  regrets_mat.mean(axis=0),
        "ep_regret_std":   regrets_mat.std(axis=0),
        "total_reward_mean": rewards_mat.sum(axis=1).mean(),
        "total_reward_std":  rewards_mat.sum(axis=1).std(),
        "total_regret_mean": regrets_mat.sum(axis=1).mean(),
        "total_regret_std":  regrets_mat.sum(axis=1).std(),
    }


# ── Print summary ─────────────────────────────────────────────────────────────
print(f"\n{'Agent':<28} {'Reward (mean±std)':>22} {'Regret (mean±std)':>22}")
print("-" * 75)
for name in agent_names:
    r = agg[name]["total_reward_mean"]; rs = agg[name]["total_reward_std"]
    g = agg[name]["total_regret_mean"]; gs = agg[name]["total_regret_std"]
    print(f"{name:<28} {r:>10.1f} ± {rs:<8.1f} {g:>10.2f} ± {gs:<8.2f}")


# ── Plots ─────────────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
colors = ["royalblue", "mediumorchid", "darkorange", "tomato", "seagreen"]
eps    = np.arange(N_EPISODES)

def smooth(arr, w=20):
    return np.convolve(arr, np.ones(w)/w, mode="same")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for i, (name, c) in enumerate(zip(agent_names, colors)):
    m = agg[name]["cum_reward_mean"]
    s = agg[name]["cum_reward_std"]
    axes[0].plot(m, label=name, color=c, lw=2)
    axes[0].fill_between(eps, m-s, m+s, color=c, alpha=0.12)

    m = agg[name]["cum_regret_mean"]
    s = agg[name]["cum_regret_std"]
    axes[1].plot(m, label=name, color=c, lw=2)
    axes[1].fill_between(eps, m-s, m+s, color=c, alpha=0.12)

    m = smooth(agg[name]["ep_regret_mean"])
    s = smooth(agg[name]["ep_regret_std"])
    axes[2].plot(m, label=name, color=c, lw=2)
    axes[2].fill_between(eps, m-s, m+s, color=c, alpha=0.12)

for ax in axes:
    for se in shift_episodes:
        ax.axvline(se, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlabel("Episode")

axes[0].set_title(f"Cumulative Reward (mean ± std, {N_SEEDS} seeds)", fontsize=12)
axes[1].set_title(f"Cumulative Regret (mean ± std, {N_SEEDS} seeds)", fontsize=12)
axes[2].set_title("Per-Episode Regret — Recovery After Shifts", fontsize=12)

plt.suptitle(
    f"Bayesian Bandit Ablation — {N_SEEDS} Seeds × {N_EPISODES} Episodes\n"
    "(dashed = distribution shifts)",
    fontsize=13, y=1.02
)
plt.tight_layout()
plt.savefig("results/comparison.png", dpi=150, bbox_inches="tight")
print("\nMain plot saved to results/comparison.png")


# ── PGM + ArviZ trace plots ───────────────────────────────────────────────────
print("\n=== Fitting Hierarchical Bayesian Network (PGM) ===")
from models.reward_pgm import build_bandit_pgm, plot_pgm_posteriors

bb_agent = final_agents["Beta-Bernoulli TS"]
all_arms, all_rewards = [], []
for a in range(N_ARMS):
    for r in bb_agent.history[a]:
        all_arms.append(a)
        all_rewards.append(int(r))

pgm_model, pgm_trace = build_bandit_pgm(
    arm_ids=np.array(all_arms),
    rewards=np.array(all_rewards),
    n_arms=N_ARMS
)
plot_pgm_posteriors(pgm_trace, N_ARMS, initial_means)

axes_arviz = az.plot_trace(pgm_trace, var_names=["mu_global", "kappa_global", "theta"],
                           combined=False)
plt.suptitle("NUTS Chain Convergence — Hierarchical Bayesian Network", fontsize=12)
plt.tight_layout()
plt.savefig("results/pgm_trace.png", dpi=150, bbox_inches="tight")
print("ArviZ trace plot saved to results/pgm_trace.png")
plt.close("all")


# ── Posterior evolution GIF ───────────────────────────────────────────────────
print("\n=== Generating Posterior Evolution GIF ===")
snapshots = bb_agent.posterior_snapshots
if len(snapshots) > 0:
    x = np.linspace(0, 1, 200)
    fig_gif, ax_gif = plt.subplots(figsize=(8, 4))

    def animate(i):
        ax_gif.clear()
        alphas, betas_p = snapshots[i]
        for a in range(N_ARMS):
            y = beta_dist.pdf(x, alphas[a], betas_p[a])
            ax_gif.plot(x, y, label=f"Arm {a} (p={initial_means[a]})",
                        color=colors[a], lw=2)
            ax_gif.axvline(initial_means[a], color=colors[a],
                           linestyle=":", alpha=0.6)
        ax_gif.set_xlim(0, 1); ax_gif.set_ylim(0, 15)
        ax_gif.set_title(f"Beta Posterior Evolution — Episode {i*10}", fontsize=12)
        ax_gif.set_xlabel("θ (reward probability)")
        ax_gif.set_ylabel("Density")
        ax_gif.legend(fontsize=7, loc="upper left")
        ax_gif.grid(alpha=0.3)

    ani = animation.FuncAnimation(fig_gif, animate, frames=len(snapshots),
                                  interval=120, repeat=True)
    ani.save("results/posterior_evolution.gif", writer="pillow", fps=10)
    print("GIF saved to results/posterior_evolution.gif")
    plt.close("all")


# ── CSV ───────────────────────────────────────────────────────────────────────
with open("results/metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    header = ["episode"]
    for name in agent_names:
        slug = name.lower().replace(" ", "_").replace("-", "")
        header += [f"{slug}_reward_mean", f"{slug}_reward_std",
                   f"{slug}_regret_mean", f"{slug}_regret_std"]
    writer.writerow(header)
    for i in range(N_EPISODES):
        row = [i+1]
        for name in agent_names:
            row += [
                agg[name]["cum_reward_mean"][i],
                agg[name]["cum_reward_std"][i],
                agg[name]["cum_regret_mean"][i],
                agg[name]["cum_regret_std"][i],
            ]
        writer.writerow(row)
print("Metrics saved to results/metrics.csv")
print("\nDone! Results in results/")
