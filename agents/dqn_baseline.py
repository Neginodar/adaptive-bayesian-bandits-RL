import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),      nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int,
                 lr: float = 1e-3, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 10_000, batch_size: int = 64):
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.device        = torch.device("cpu")  # CPU is sufficient

        self.q_net     = QNetwork(obs_dim, n_actions).to(self.device)
        self.target    = QNetwork(obs_dim, n_actions).to(self.device)
        self.target.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory    = deque(maxlen=buffer_size)
        self.steps     = 0

    def choose_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.q_net(s).argmax(dim=1).item())

    def store(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s2, d = zip(*batch)
        s   = torch.FloatTensor(np.array(s)).to(self.device)
        a   = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r   = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        s2  = torch.FloatTensor(np.array(s2)).to(self.device)
        d   = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        q_vals = self.q_net(s).gather(1, a)
        with torch.no_grad():
            next_q = self.target(s2).max(1)[0].unsqueeze(1)
            target = r + self.gamma * next_q * (1 - d)

        loss = nn.MSELoss()(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.steps  += 1
        if self.steps % 100 == 0:
            self.target.load_state_dict(self.q_net.state_dict())
