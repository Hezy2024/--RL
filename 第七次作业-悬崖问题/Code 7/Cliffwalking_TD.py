from Cliffwalking_env import ENV
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Latex
from IPython.display import display

# 实现ε-greedy的策略
def choose_action_by_epsilon_greedy(state, Q, epsilon=0.2):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice([0, 1, 2, 3])  # 随机选择一个动作
    else:
        i, j = state
        return np.argmax(Q[i, j])  # 选择Q值最大的动作

# 实现SARSA
def sarsa(env, Q, alpha=0.5, gamma=1, epsilon=0.1):
    s = env.START
    a = choose_action_by_epsilon_greedy(s, Q, epsilon)
    g = 0
    while s != env.GOAL:
        next_S, r, term = env.step(s, a)
        next_a = choose_action_by_epsilon_greedy(next_S, Q, epsilon)
        i, j = s
        next_i, next_j = next_S
        Q[i, j, a] = Q[i, j, a] + alpha * (r + gamma * Q[next_i, next_j, next_a] - Q[i, j, a])
        s = next_S
        a = next_a
        g += r
    return g

# 实现Qlearning
def q_learning(env, Q, alpha=0.5, gamma=1, epsilon=0.1):
    s = env.START
    g = 0
    while s != env.GOAL:
        a = choose_action_by_epsilon_greedy(s, Q, epsilon)
        next_S, r, term = env.step(s, a)
        i, j = s
        next_i, next_j = next_S
        Q[i, j, a] = Q[i, j, a] + alpha * (r + gamma * np.max(Q[next_i, next_j]) - Q[i, j, a])
        s = next_S
        g += r
    return g

def plot_sarsa(episodes=500, N=50):
    mean_returns = np.zeros(episodes)
    env = ENV()
    for n in range(N):
        Q = np.zeros((env.WORLD_HEIGHT, env.WORLD_WIDTH, 4))
        for i in range(episodes):
            mean_returns[i] += sarsa(env, Q, 0.5)
    mean_returns /= N
    print(mean_returns)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.ylim([-100, 0])
    plt.plot(mean_returns)
    plt.show()
    return mean_returns

def plot_q_learning(episodes=500, N=50):
    mean_returns = np.zeros(episodes)
    env = ENV()
    for n in range(N):
        Q = np.zeros((env.WORLD_HEIGHT, env.WORLD_WIDTH, 4))
        for i in range(episodes):
            mean_returns[i] += q_learning(env, Q)
    mean_returns /= N
    plt.plot(mean_returns)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.ylim([-100, 0])
    return mean_returns

def print_optimal_policy(Q, env):
    ACTION_LATEX_LISTS = [r'\uparrow', r'\downarrow', r'\leftarrow', r'\rightarrow']
    policy_list = []
    for i in range(0, env.WORLD_HEIGHT):
        one_column_list = []
        for j in range(0, env.WORLD_WIDTH):
            if [i, j] == env.GOAL:
                one_column_list.append('G')
                continue
            if i == 3 and 1 <= j <= 10:
                one_column_list.append('\square')
                continue
            max_a = np.argmax(Q[i, j])
            one_column_list.append(ACTION_LATEX_LISTS[max_a])
        one_column_str = '&'.join(one_column_list)
        policy_list.append(one_column_str)
    policy_str = r'$$\begin{bmatrix}' + r'\\'.join(policy_list) + r'\end{bmatrix}$$'
    display(Latex(policy_str))

def get_optimal_path(Q, env):
    path = []
    s = env.START
    while s != env.GOAL:
        path.append(s)
        i, j = s
        a = np.argmax(Q[i, j])
        next_S, _, _ = env.step(s, a)
        s = next_S
    path.append(env.GOAL)
    return path

def plot_policy_using_matplotlib(Q, env, title):
    fig, ax = plt.subplots()
    ax.set_xlim(0, env.WORLD_WIDTH)
    ax.set_ylim(0, env.WORLD_HEIGHT)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0, env.WORLD_WIDTH + 1, 1))
    ax.set_yticks(np.arange(0, env.WORLD_HEIGHT + 1, 1))
    ax.grid(True)

    ACTION_DIRECTIONS = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # up, down, left, right
    path = get_optimal_path(Q, env)
    for i in range(env.WORLD_HEIGHT):
        for j in range(env.WORLD_WIDTH):
            if [i, j] == env.GOAL:
                ax.text(j + 0.5, env.WORLD_HEIGHT - i - 0.5, 'G', ha='center', va='center')
            elif i == 3 and 1 <= j <= 10:
                ax.text(j + 0.5, env.WORLD_HEIGHT - i - 0.5, 'X', ha='center', va='center')
            elif [i, j] in path and [i, j] != env.GOAL:
                max_a = np.argmax(Q[i, j])
                dx, dy = ACTION_DIRECTIONS[max_a]
                ax.arrow(j + 0.5, env.WORLD_HEIGHT - i - 0.5, dx * 0.3, -dy * 0.3, head_width=0.1, head_length=0.1, fc='k', ec='k')

    ax.set_title(title)
    plt.show()

def plot_sarsa_policy(episodes=1000):  # 要找到最优策略可能需要训练更多的片段数
    env = ENV()
    Q = np.zeros((env.WORLD_HEIGHT, env.WORLD_WIDTH, 4))
    for i in range(episodes):
        sarsa(env, Q)
    plot_policy_using_matplotlib(Q, env, "SARSA Optimal Policy")

def plot_q_policy(episodes=1000):  # 要找到最优策略可能需要训练更多的片段数
    env = ENV()
    Q = np.zeros((env.WORLD_HEIGHT, env.WORLD_WIDTH, 4))
    for i in range(episodes):
        q_learning(env, Q)
    plot_policy_using_matplotlib(Q, env, "Q-Learning Optimal Policy")

if __name__ == "__main__":
    env = ENV()
    sarsa_returns = plot_sarsa()
    plot_sarsa_policy()
    q_returns = plot_q_learning()
    plot_q_policy()
    plt.plot(sarsa_returns, label='sarsa')
    plt.plot(q_returns, label='Q-learning')
    plt.ylim([-100, 0])
    plt.legend()
    plt.show()

