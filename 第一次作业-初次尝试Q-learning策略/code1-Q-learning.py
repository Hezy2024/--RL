#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import random

BOARD_LEN = 3
EPSILON = 0.1  # 探索率
ALPHA = 0.5    # 学习率
GAMMA = 0.9    # 折扣因子
params = [
    (0.1, 0.5, 0.9),
    (0.2, 0.5, 0.9),
    (0.1, 0.3, 0.9),
    (0.1, 0.5, 0.8),
    (0.1, 0.5, 0.7),
    # 添加更多的参数组合
]
class TicTacToeEnv(object):
    def __init__(self): #用数值的方式表示状态、动作、奖励(+1/0/-1 区分胜/平/负)
        self.data = np.zeros((BOARD_LEN, BOARD_LEN))  # data 表示棋盘当前状态，1和-1分别表示x和o，0表示空位
        self.winner = None  # 1/0/-1表示玩家一胜/平局/玩家二胜，None表示未分出胜负
        self.terminal = False  # true表示游戏结束
        self.current_player = 1  # 当前正在下棋的人是玩家1还是-1

    def reset(self):
        # 游戏重新开始，返回状态
        self.data = np.zeros((BOARD_LEN, BOARD_LEN))
        self.winner = None
        self.terminal = False
        state = self.getState()
        return state

    def getState(self):
        # 注意到很多时候，存储数据不等同与状态，状态的定义可以有很多种，比如将棋的位置作一些哈希编码等
        # 这里直接返回data数据作为状态
        return self.data

    def getReward(self):
        """Return (reward_1, reward_2)
        """
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
        if self.current_player == 1:
            self.current_player = -1
        else:
            self.current_player = 1

    def checkState(self):
        # 每次有人下棋，都要检查游戏是否结束，如何检查呢？思考
        # 从而更新self.terminal和self.winner
        # ----------------------------------
        # 实现自己的代码
        #检查是否有一方胜利或者和棋：即行、列、对角线是否全为1/-1,若不存在，则判断是否1/-1的数量为9个
        if self.terminal != False:
            return self.terminal
        
        results=[]  #参考网上的一种方法，使用一个数组即可实现上述判断

        #首先，将行、列、对角线的和放入数组中：
        for i in range(0, BOARD_LEN):
            results.append(np.sum(self.data[i, :]))
    
        for i in range(0, BOARD_LEN):
            results.append(np.sum(self.data[:, i]))

        results.append(0)
        for i in range(0, BOARD_LEN):
            results[-1] += self.data[i, i]   
        results.append(0)
        for i in range(0, BOARD_LEN):
            results[-1] += self.data[i, BOARD_LEN - 1 - i]

        #至此，results内有8个数，分别是三行、三列、两个对角线的和，现在只需要判断这里面有没有3或者-3即可。

        for result in results:
            if result == 3:
                self.winner = 1
                self.terminal = True
                return self.terminal
            if result == -3:
                self.winner = -1
                self.terminal = True
                return self.terminal

        sum = np.sum(np.abs(self.data)) #通过取绝对值再相加，判断是否有9个棋子
        if sum == BOARD_LEN * BOARD_LEN:
            self.winner = 0
            self.terminal = True
            return self.terminal

        self.terminal= False
        return self.terminal
        #check完毕

        # ----------------------------------
        

    def step(self, action):
        """action: is a tuple or list [x, y]
        Return:
            state, reward, terminal
        """
        # ----------------------------------
        value = 1 if self.current_player == 1 else -1
        self.data[action[0], action[1]] = value
        self.checkState()
        next_state = self.getState()
        reward = self.getReward()
        self.switchPlayer()
        return next_state, reward, self.terminal
        # ----------------------------------
    #我加上一个可视化的过程    
    def print(self):
        for i in range(0, BOARD_LEN):
            print('-------------')
            out = '| '
            for j in range(0, BOARD_LEN):
                if self.data[i, j] == 1:
                    token = '*'
                if self.data[i, j] == 0:
                    token = '0'
                if self.data[i, j] == -1:
                    token = 'x'
                out += token + ' | '
            print(out)
        print('-------------')


class QAgent(object):
    def __init__(self):
        self.q_table = {}  # Q表
        self.state_size = BOARD_LEN * BOARD_LEN  # 状态空间
        self.action_size = BOARD_LEN * BOARD_LEN  # 动作空间

    def get_state_key(self, state):
        return str(state.flatten())  # 将状态转换为字符串作为键

    def policy(self, state):
        state_key = self.get_state_key(state)
        if random.uniform(0, 1) < EPSILON:  # 探索，有着一定的随机性
            available_actions = np.argwhere(state == 0)
            action = random.choice(available_actions)
        else:  # 利用
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_size)
            action = np.argmax(self.q_table[state_key])  # 选择最大Q值的动作
            action = divmod(action, BOARD_LEN)  # 将一维索引转换为二维坐标
        return action

    def update_q_value(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)

        action_index = action[0] * BOARD_LEN + action[1]
        
        # Q值更新公式
        self.q_table[state_key][action_index] += ALPHA * (
            reward + GAMMA * np.max(self.q_table[next_state_key]) - self.q_table[state_key][action_index]
        )
        #我觉得Q-learning的核心就在于这个更新公式，
        #其他的是用来服务或者说是撑起这个方法的结构，如果想要优化，从这个下手
        #另外，我觉得还没有完全搞懂这个公式

def main():
    env = TicTacToeEnv()
    agent1 = QAgent()
    agent2 = QAgent()
    num_episodes = 1000

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
                
                winner_str = 'Player1' if winner == 1 else 'Player2' if winner == -1 else 'Draw'
                print(f'Episode {episode + 1}/{num_episodes}: Winner: {winner_str}')
                break

            agent1.update_q_value(state, action, reward, next_state) if current_player == 1 else agent2.update_q_value(state, action, reward, next_state)
            state = next_state

        # 进度条
        progress = (episode + 1) / num_episodes * 100
        sys.stdout.write(f'\rTraining Progress: {progress:.2f}%')
        sys.stdout.flush()

    print("\nTraining complete.")
    print(f'Player 1 Wins: {wins_player1}, Player 2 Wins: {wins_player2}, Draws: {draws}')

if __name__ == "__main__":
    main()

#训练完毕，大概使用了五分钟不到