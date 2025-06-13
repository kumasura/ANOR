# RL script for flight rerouting using Q-learning
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class FlightReroutingEnv(gym.Env):
    """Simple environment demonstrating rerouting decisions."""

    def __init__(self, schedule_path, disruption_prob=0.3, delay_minutes=60):
        super().__init__()
        self.schedule = pd.read_excel(schedule_path)
        self.num_flights = len(self.schedule)
        self.disruption_prob = disruption_prob
        self.delay_minutes = delay_minutes

        # Actions: 0 operate, 1 delay, 2 cancel
        self.action_space = spaces.Discrete(3)
        # Observation: index of current flight (0..num_flights). We use num_flights to represent terminal state.
        self.observation_space = spaces.Discrete(self.num_flights + 1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = 0
        self.done = False
        return self.current_idx, {}

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done")

        row = self.schedule.iloc[self.current_idx]
        disrupted = np.random.rand() < self.disruption_prob

        delay = 0
        reward = 0.0
        if disrupted:
            # flight cannot depart as scheduled
            if action == 0:  # try to operate
                reward = -50.0  # penalty for failed operation
            elif action == 1:  # delay
                delay = self.delay_minutes
                reward = -delay
            elif action == 2:  # cancel
                reward = -100.0
        else:
            if action == 0:
                reward = 0.0  # on time
            elif action == 1:
                delay = self.delay_minutes
                reward = -delay
            elif action == 2:
                reward = -100.0

        self.current_idx += 1
        terminated = self.current_idx >= self.num_flights
        obs = self.num_flights if terminated else self.current_idx
        self.done = terminated
        return obs, reward, terminated, False, {}

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.Q = np.zeros((state_size, action_size))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_size = action_size

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error


def train(env, episodes=1000):
    agent = QLearningAgent(env.observation_space.n, env.action_space.n)
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, _, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            done = terminated
    return agent


def main():
    env = FlightReroutingEnv('flight_schedule_new.xlsx')
    agent = train(env)
    print("Trained Q-table:\n", agent.Q)

if __name__ == '__main__':
    main()
