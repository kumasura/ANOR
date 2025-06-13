import numpy as np
from rerouting_rl import FlightReroutingEnv


class DQN:
    def __init__(self, state_bins, action_size, hidden_size=64, lr=0.01, gamma=0.95):
        self.state_bins = state_bins
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        input_size = len(state_bins)
        self.w1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, action_size) * 0.01
        self.b2 = np.zeros(action_size)

    def _forward(self, x):
        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(z1, 0)
        q = h1 @ self.w2 + self.b2
        return q, h1, z1

    def predict(self, x):
        q, _, _ = self._forward(x)
        return q

    def update(self, batch):
        for state, action, reward, next_state, done in batch:
            q, h1, z1 = self._forward(state)
            target = reward
            if not done:
                next_q = self.predict(next_state)
                target += self.gamma * np.max(next_q)
            dq = np.zeros_like(q)
            dq[action] = q[action] - target
            grad_w2 = np.outer(h1, dq)
            grad_b2 = dq
            grad_h1 = dq @ self.w2.T
            grad_z1 = grad_h1 * (z1 > 0)
            grad_w1 = np.outer(state, grad_z1)
            grad_b1 = grad_z1
            self.w2 -= self.lr * grad_w2
            self.b2 -= self.lr * grad_b2
            self.w1 -= self.lr * grad_w1
            self.b1 -= self.lr * grad_b1


def normalize(state, bins):
    return np.array(state) / (bins - 1)


def train_dqn(env, episodes=200, batch_size=32, buffer_limit=10000, epsilon=1.0, epsilon_decay=0.995):
    dqn = DQN(env.obs_bins, env.action_space.n)
    replay = []
    for ep in range(episodes):
        state, _ = env.reset()
        state = normalize(state, env.obs_bins)
        done = False
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                q = dqn.predict(state)
                action = int(np.argmax(q))
            next_state, reward, terminated, _, _ = env.step(action)
            next_state_n = normalize(next_state, env.obs_bins)
            replay.append((state, action, reward, next_state_n, terminated))
            if len(replay) > buffer_limit:
                replay.pop(0)
            if len(replay) >= batch_size:
                batch_idx = np.random.choice(len(replay), batch_size, replace=False)
                batch = [replay[i] for i in batch_idx]
                dqn.update(batch)
            state = next_state_n
            done = terminated
        if epsilon > 0.1:
            epsilon *= epsilon_decay
    return dqn


def evaluate_dqn(env, dqn):
    action_names = [
        "Change path",
        "Swap aircraft",
        "Cancel flight",
        "Adjust altitude",
        "Divert",
        "Wait",
    ]
    state, _ = env.reset()
    state_n = normalize(state, env.obs_bins)
    done = False
    total_reward = 0
    while not done:
        action = int(np.argmax(dqn.predict(state_n)))
        next_state, reward, terminated, _, _ = env.step(action)
        print(
            f"Flight {state[0]} -> action: {action_names[action]} | state: {state[1:]} | reward: {reward:.1f}"
        )
        total_reward += reward
        state = next_state
        state_n = normalize(state, env.obs_bins)
        done = terminated
    print("Total reward:", total_reward)


if __name__ == "__main__":
    env = FlightReroutingEnv("flight_schedule_new.xlsx")
    agent = train_dqn(env, episodes=100)
    print("\nPolicy rollout:\n")
    evaluate_dqn(env, agent)
