import numpy as np
from rerouting_rl import FlightReroutingEnv
from dqn_train import train_dqn, normalize


def generate_schedule(env, dqn, epsilon=0.05):
    """Generate a schedule using an epsilon-greedy policy with the DQN."""
    state, _ = env.reset()
    state_n = normalize(state, env.obs_bins)
    done = False
    actions = []
    total_reward = 0.0
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(dqn.predict(state_n)))
        next_state, reward, terminated, _, _ = env.step(action)
        actions.append(action)
        total_reward += reward
        state = next_state
        state_n = normalize(state, env.obs_bins)
        done = terminated
    return actions, total_reward


def column_generation(schedule_path, iterations=20):
    """Run column generation enhanced with a DQN agent."""
    env = FlightReroutingEnv(schedule_path)
    dqn = train_dqn(env)

    columns = []
    best_actions = None
    best_cost = float("inf")

    for _ in range(iterations):
        actions, reward = generate_schedule(env, dqn)
        cost = -reward
        columns.append((actions, cost))
        if cost < best_cost:
            best_cost = cost
            best_actions = actions

    return columns, best_actions, best_cost


if __name__ == "__main__":
    cols, best_sched, best_cost = column_generation("flight_schedule_new.xlsx")
    print("Generated columns:", len(cols))
    print("Best schedule cost:", best_cost)
    print("Best schedule actions:", best_sched)
