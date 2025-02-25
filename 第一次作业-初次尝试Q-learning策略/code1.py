#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import random

BOARD_LEN = 3


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


class RandAgent(object):
    def policy(self, state):
        """
        Return: action
        """
        #最简单的就是随机，但是这样就没有学习的过程了
        #所以我先尝试一下随机的做法，然后再去参考一下怎么做到让棋手针对不同状态做出最优解
        # ----------------------------------
        #随机：
        # 获取所有可用的动作（空位）
        available_actions = np.argwhere(state == 0)  # 找到所有值为0的索引
        # 随机选择一个可用的动作
        action = random.choice(available_actions)
        return tuple(action)  # 返回为元组形式
        
        # ----------------------------------
        
      

#纯随机的main()
def main():
    env = TicTacToeEnv()
    agent1 = RandAgent()
    agent2 = RandAgent()
    state = env.reset()

    # 这里给出了一次运行的代码参考
    # 你可以按照自己的想法实现
    # 多次实验，计算在双方随机策略下，先手胜/平/负的概率
    while 1:
        current_player = env.getCurrentPlayer()
        if current_player == 1:
            action = agent1.policy(state)
        else:
            action = agent2.policy(state)
        next_state, reward, terminal = env.step(action)
        env.print()
        print(next_state)
        if terminal:
            winner = 'Player1' if env.getWinner() == 1 else 'Player2'
            print('Winner: {}'.format(winner))
            break
        state = next_state


if __name__ == "__main__":
    main()
