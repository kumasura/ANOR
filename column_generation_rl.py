import numpy as np
from rerouting_rl import FlightReroutingEnv, train


def generate_schedule(env, agent, epsilon=0.05):
    """Generate a schedule (action for each flight) using an epsilon-greedy policy."""
    state, _ = env.reset()
    done = False
    actions = []
    total_reward = 0.0
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = agent.choose_action(state)
        next_state, reward, terminated, _, _ = env.step(action)
        actions.append(action)
        total_reward += reward
        state = next_state
        done = terminated
    return actions, total_reward


def column_generation(schedule_path, iterations=20):
    """Run column generation enhanced by an RL agent."""
    env = FlightReroutingEnv(schedule_path)
    agent = train(env)

    columns = []
    best_actions = None
    best_cost = float("inf")

    for _ in range(iterations):
        actions, reward = generate_schedule(env, agent)
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
