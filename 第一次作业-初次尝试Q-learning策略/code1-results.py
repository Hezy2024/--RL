import numpy as np
import random
import sys
import pandas as pd

# Q-learning 参数的组合
params = [
    (0.1, 0.5, 0.9),
    (0.2, 0.5, 0.9),
    (0.1, 0.3, 0.9),
    (0.1, 0.5, 0.8),
    (0.1, 0.5, 0.7),
    # 添加更多的参数组合
]

BOARD_LEN = 3

class TicTacToeEnv:
    def __init__(self):
        self.data = np.zeros((BOARD_LEN, BOARD_LEN))
        self.winner = None
        self.terminal = False
        self.current_player = 1

    def reset(self):
        self.data = np.zeros((BOARD_LEN, BOARD_LEN))
        self.winner = None
        self.terminal = False
        return self.getState()

    def getState(self):
        return self.data

    def getReward(self):
        if self.terminal:
            if self.winner == 1:
                return 1, -1
            elif self.winner == -1:
                return -1, 1
        return 0, 0

    def getCurrentPlayer(self):
        return self.current_player

    def getWinner(self):
        return self.winner

    def switchPlayer(self):
        self.current_player = -self.current_player

    def checkState(self):
        if self.terminal:
            return self.terminal
        
        results = []
        for i in range(BOARD_LEN):
            results.append(np.sum(self.data[i, :]))
        for i in range(BOARD_LEN):
            results.append(np.sum(self.data[:, i]))
        results.append(np.sum(self.data.diagonal()))
        results.append(np.sum(np.fliplr(self.data).diagonal()))

        for result in results:
            if result == 3:
                self.winner = 1
                self.terminal = True
                return self.terminal
            if result == -3:
                self.winner = -1
                self.terminal = True
                return self.terminal

        if np.all(self.data != 0):
            self.winner = 0
            self.terminal = True
            return self.terminal

        self.terminal = False
        return self.terminal

    def step(self, action):
        value = 1 if self.current_player == 1 else -1
        self.data[action[0], action[1]] = value
        self.checkState()
        next_state = self.getState()
        reward = self.getReward()
        self.switchPlayer()
        return next_state, reward, self.terminal

class QAgent:
    def __init__(self, epsilon, alpha, gamma):
        self.q_table = {}
        self.state_size = BOARD_LEN * BOARD_LEN
        self.action_size = BOARD_LEN * BOARD_LEN
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def get_state_key(self, state):
        return str(state.flatten())

    def policy(self, state):
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if random.uniform(0, 1) < self.epsilon:
            available_actions = np.argwhere(state == 0)
            action = random.choice(available_actions)
        else:
            action = np.argmax(self.q_table[state_key])
            action = divmod(action, BOARD_LEN)
        return action

    def update_q_value(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)

        action_index = action[0] * BOARD_LEN + action[1]
        self.q_table[state_key][action_index] += self.alpha * (
            reward[0] + self.gamma * np.max(self.q_table[next_state_key]) - self.q_table[state_key][action_index]
        )

def main():
    num_episodes = 1000
    results = []

    for epsilon, alpha, gamma in params:
        env = TicTacToeEnv()
        agent1 = QAgent(epsilon, alpha, gamma)
        agent2 = QAgent(epsilon, alpha, gamma)

        # 胜利计数
        wins_player1 = 0
        wins_player2 = 0
        draws = 0

        for episode in range(num_episodes):
            state = env.reset()
            while True:
                current_player = env.getCurrentPlayer()
                action = agent1.policy(state) if current_player == 1 else agent2.policy(state)
                next_state, reward, terminal = env.step(action)

                if terminal:
                    winner = env.getWinner()
                    if winner == 1:
                        wins_player1 += 1
                    elif winner == -1:
                        wins_player2 += 1
                    else:
                        draws += 1
                    break
                
                agent1.update_q_value(state, action, reward, next_state) if current_player == 1 else agent2.update_q_value(state, action, reward, next_state)
                state = next_state

            # 进度条
            progress = (episode + 1) / num_episodes * 100
            sys.stdout.write(f'\rTraining Progress: {progress:.2f}%')
            sys.stdout.flush()

        # 记录结果
        results.append({
            'Epsilon': epsilon,
            'Alpha': alpha,
            'Gamma': gamma,
            'Player 1 Wins': wins_player1,
            'Player 2 Wins': wins_player2,
            'Draws': draws,
        })

    print("\nTraining complete.")
    
    # 将结果保存到 Excel
    df = pd.DataFrame(results)
    df.to_excel('code1_results.xlsx', index=False)
    print("Results saved to 'code1_results.xlsx'.")

if __name__ == "__main__":
    main()