import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class FlightReroutingEnv(gym.Env):

    """Environment using schedule data with simple simulated disruptions."""
    def __init__(self, schedule_path, disruption_prob=0.3):

        super().__init__()
        self.schedule = pd.read_excel(schedule_path)
        self.num_flights = len(self.schedule)
        self.disruption_prob = disruption_prob


        # Observation consists of: flight index, fuel level, weather, traffic,
        # alternate airports, other aircraft proximity
        self.obs_bins = np.array([
            self.num_flights + 1,  # flight index including terminal state
            5,  # fuel level bins
            5,  # weather bins
            5,  # traffic bins
            4,  # number of alternate airports
            5,  # other aircraft proximity bins
        ])
        self.observation_space = spaces.MultiDiscrete(self.obs_bins)

        # Actions follow the README: change path, swap aircraft, cancel, adjust
        # altitude, divert, wait for conditions to improve
        self.action_space = spaces.Discrete(6)

    def _random_state(self):
        return [
            np.random.randint(self.obs_bins[1]),
            np.random.randint(self.obs_bins[2]),
            np.random.randint(self.obs_bins[3]),
            np.random.randint(self.obs_bins[4]),
            np.random.randint(self.obs_bins[5]),
        ]


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = 0

        self.state = self._random_state()
        return tuple([self.current_idx] + self.state), {}

    def step(self, action):
        if self.current_idx >= self.num_flights:
            raise RuntimeError("Episode is done")

        disruption = np.random.rand() < self.disruption_prob
        fuel, weather, traffic, airports, other = self.state

        # Simple cost model
        fuel_cost = 10 + 5 * weather + 5 * traffic
        delay_penalty = 20 if action == 5 else 0
        swap_penalty = 50 if action == 1 else 0
        cancel_penalty = 200 if action == 2 else 0
        reroute_penalty = 30 if action in (0, 3, 4) else 0
        if not disruption and action == 5:
            delay_penalty += 20  # unnecessary waiting

        reward = -(fuel_cost + delay_penalty + swap_penalty + cancel_penalty + reroute_penalty)

        # Advance to next flight and generate new state
        self.current_idx += 1
        terminated = self.current_idx >= self.num_flights
        self.state = self._random_state() if not terminated else [0] * 5
        obs = tuple([self.num_flights] + self.state) if terminated else tuple([self.current_idx] + self.state)
        return obs, reward, terminated, False, {}


class QLearningAgent:
    def __init__(self, nvec, action_size, alpha=0.1, gamma=0.95, epsilon=0.1):
        state_space = int(np.prod(nvec))
        self.Q = np.zeros((state_space, action_size))
        self.nvec = nvec
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_size = action_size


    def _state_index(self, state):
        return np.ravel_multi_index(state, self.nvec)

    def choose_action(self, state):
        idx = self._state_index(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        return int(np.argmax(self.Q[idx]))

    def update(self, state, action, reward, next_state):
        idx = self._state_index(state)
        next_idx = self._state_index(next_state)
        best_next = np.max(self.Q[next_idx])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[idx, action]
        self.Q[idx, action] += self.alpha * td_error


def train(env, episodes=200):
    agent = QLearningAgent(env.obs_bins, env.action_space.n)
    for _ in range(episodes):

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

    print("Q-table shape:", agent.Q.shape)
    # show small portion
    print(agent.Q[:5])


if __name__ == '__main__':
    main()
