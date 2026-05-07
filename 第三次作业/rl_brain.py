from __future__ import annotations

import random
from collections import defaultdict
from typing import DefaultDict, List, Tuple


State = Tuple[int, int]


class QLearningAgent:
    """Q-Learning 智能体，维护 Q-table 并使用 epsilon-greedy 策略。"""

    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ) -> None:
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: DefaultDict[State, List[float]] = defaultdict(
            lambda: [0.0 for _ in range(self.n_actions)]
        )

    def choose_action(self, state: State) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return self.greedy_action(state)

    def greedy_action(self, state: State) -> int:
        q_values = self.q_table[state]
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    def learn(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        done: bool,
    ) -> None:
        current_q = self.q_table[state][action]
        next_max_q = 0.0 if done else max(self.q_table[next_state])
        target = reward + self.gamma * next_max_q
        self.q_table[state][action] += self.alpha * (target - current_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
