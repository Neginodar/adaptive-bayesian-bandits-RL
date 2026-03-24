import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
from scipy.stats import beta as beta_dist
import csv

# ── Config ────────────────────────────────────────────────────────────────────
N_ARMS     = 5
N_EPISODES = 500
TRUE_PROBS = [0.1, 0.3, 0.5, 0.7, 0.9]  # true reward prob per arm
WINDOW     = 20

# ── Bayesian Thompson Sampling Agent ─────────────────────────────────────────
class ThompsonSamplingAgent:
    """Beta-Bernoulli Thompson Sampling — the canonical Bayesian bandit agent."""
    def __init__(self, n_arms):
        self.n_arms  = n_arms
        self.alphas  = np.ones(n_arms)   # Beta(alpha, beta) posterior
        self.betas_p = np.ones(n_arms)
        self.traces  = {}
        self.history = {a: [] for a in range(n_arms)}

    def choose(self):
        samples = np.random.beta(self.alphas, self.betas_p)
        return int(np.argmax(samples))

    def update(self, arm, reward):
        self.alphas[arm]  += reward
        self.betas_p[arm] += (1 - reward)
        self.history[arm].append(reward)

    def run_mcmc(self, episode):
        """Full PyMC posterior — validates Beta-Bernoulli analytically."""
        for arm in range(self.n_arms):
            h = self.history[arm]
            if len(h) < 5:
                continue
            n = len(h); s = int(sum(h))
            with pm.Model():
                theta = pm.Beta("theta", alpha=1.0, beta=1.0)
                pm.Binomial("obs", n=n, p=theta, observed=s)
                trace = pm.sample(500, tune=500, chains=2,
                                  progressbar=False, return_inferencedata=True)
            self.traces[(episode, arm)] = trace

    def posterior_entropy(self):
        return float(np.mean([
            beta_dist.entropy(a, b)
            for a, b in zip(self.alphas, self.betas_p)
        ]))

# ── Epsilon-Greedy Baseline ───────────────────────────────────────────────────
class EpsilonGreedyAgent:
    """Standard epsilon-greedy bandit baseline."""
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

# ── Bandit Environment ────────────────────────────────────────────────────────
def pull(arm):
    return 1 if np.random.random() < TRUE_PROBS[arm] else 0

# ── Run experiments ───────────────────────────────────────────────────────────
ts_agent  = ThompsonSamplingAgent(N_ARMS)
eg_agent  = EpsilonGreedyAgent(N_ARMS, epsilon=0.1)

ts_rewards, eg_rewards, ts_regrets, eg_regrets, entropies = [], [], [], [], []
optimal_prob = max(TRUE_PROBS)

print("=== Running Thompson Sampling (Bayesian) ===")
for ep in range(N_EPISODES):
    arm    = ts_agent.choose()
    reward = pull(arm)
    ts_agent.update(arm, reward)
    ts_rewards.append(reward)
    ts_regrets.append(optimal_prob - TRUE_PROBS[arm])
    entropies.append(ts_agent.posterior_entropy())
    if (ep + 1) % 100 == 0:
        print(f"  ep {ep+1} | cumulative reward={sum(ts_rewards)} | "
              f"cumulative regret={sum(ts_regrets):.2f}")
    if (ep + 1) % 100 == 0:
        ts_agent.run_mcmc(ep + 1)

print("\n=== Running Epsilon-Greedy Baseline ===")
for ep in range(N_EPISODES):
    arm    = eg_agent.choose()
    reward = pull(arm)
    eg_agent.update(arm, reward)
    eg_rewards.append(reward)
    eg_regrets.append(optimal_prob - TRUE_PROBS[arm])
    if (ep + 1) % 100 == 0:
        print(f"  ep {ep+1} | cumulative reward={sum(eg_rewards)} | "
              f"cumulative regret={sum(eg_regrets):.2f}")

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\nThompson Sampling  — total reward: {sum(ts_rewards)} | "
      f"total regret: {sum(ts_regrets):.2f}")
print(f"Epsilon-Greedy     — total reward: {sum(eg_rewards)} | "
      f"total regret: {sum(eg_regrets):.2f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
def smooth(arr, w):
    return np.convolve(arr, np.ones(w)/w, mode="valid")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(np.cumsum(ts_rewards), label="Thompson Sampling", color="royalblue", lw=2)
axes[0].plot(np.cumsum(eg_rewards), label="Epsilon-Greedy",    color="tomato",    lw=2)
axes[0].set_title("Cumulative Reward", fontsize=13)
axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Total Reward")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(np.cumsum(ts_regrets), label="Thompson Sampling", color="royalblue", lw=2)
axes[1].plot(np.cumsum(eg_regrets), label="Epsilon-Greedy",    color="tomato",    lw=2)
axes[1].set_title("Cumulative Regret (lower = better)", fontsize=13)
axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Total Regret")
axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(entropies, color="mediumseagreen", lw=2)
axes[2].set_title("Posterior Entropy Decay (Thompson Sampling)", fontsize=13)
axes[2].set_xlabel("Episode"); axes[2].set_ylabel("Mean Beta Entropy")
axes[2].grid(alpha=0.3)

plt.suptitle("Bayesian Thompson Sampling vs Epsilon-Greedy — 5-Arm Bandit",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("results/comparison.png", dpi=150, bbox_inches="tight")
print("\nPlot saved to results/comparison.png")

with open("results/metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "ts_reward", "eg_reward",
                     "ts_regret", "eg_regret", "posterior_entropy"])
    for i in range(N_EPISODES):
        writer.writerow([i+1, ts_rewards[i], eg_rewards[i],
                         ts_regrets[i], eg_regrets[i], entropies[i]])
print("Metrics saved to results/metrics.csv")
