import pygame
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from yuanyang_env_mc import YuanYangEnv

class MC_RL:
    def __init__(self, yuanyang):
        # 行为值函数的初始化
        self.qvalue = np.zeros((len(yuanyang.states), len(yuanyang.actions))) * 0.1
        # 次数初始化
        # n[s,a]=1,2,3?? 求经验平均时，q(s,a)=G(s,a)/n(s,a)
        self.n = 0.01 * np.ones((len(yuanyang.states), len(yuanyang.actions)))
        self.actions = yuanyang.actions
        self.yuanyang = yuanyang
        self.gamma = yuanyang.gamma

    # 定义贪婪策略
    def greedy_policy(self, qfun, state):
        amax = qfun[state, :].argmax()
        return self.actions[amax]

    # 定义e-贪婪策略,蒙特卡罗方法，要评估的策略时e-greedy策略，产生数据的策略。
    def epsilon_greedy_policy(self, qfun, state, epsilon):
        amax = qfun[state, :].argmax()
        # 概率部分
        if np.random.uniform() < 1 - epsilon:
            # 最优动作
            return self.actions[amax]
        else:
            return self.actions[int(random.random() * len(self.actions))]

    # 找到动作所对应的序号
    def find_anum(self, a):
        for i in range(len(self.actions)):
            if a == self.actions[i]:
                return i

    def mc_learning_ei(self, num_iter):
        # 初始化Q值表和访问次数表
        self.qvalue = np.zeros((len(self.yuanyang.states), len(self.yuanyang.actions)))
        self.n = np.ones((len(self.yuanyang.states), len(self.yuanyang.actions)))

        for iter1 in range(num_iter):
            s = self.yuanyang.reset()
            a = self.epsilon_greedy_policy(self.qvalue, s, epsilon=0.1)
            done = False
            step_num = 0
            trajectory = []

            while not done:
                s_, r, t = self.yuanyang.transform(s, a)
                trajectory.append((s, a, r))
                s = s_  # 更新状态到下一个状态
                a = self.epsilon_greedy_policy(self.qvalue, s, epsilon=0.1)
                done = t
                step_num += 1
                if step_num > 100:
                    break

            G = 0
            for i in reversed(range(len(trajectory))):
                s_t, a_t, r_t = trajectory[i]
                G = r_t + self.gamma * G
                self.n[s_t, self.find_anum(a_t)] += 1
                self.qvalue[s_t, self.find_anum(a_t)] += (G - self.qvalue[s_t, self.find_anum(a_t)]) / self.n[s_t, self.find_anum(a_t)]

            if (iter1 + 1) % 1000 == 0:
                print(f"Iteration {iter1 + 1}/{num_iter} completed")

        return self.qvalue

    def mc_test(self):
        s = 0
        done = False
        step_num = 0
        path = []
        while not done and step_num < 30:
            path.append(s)
            yuanyang.path = path
            a = self.greedy_policy(self.qvalue, s)
            s_, r, t = self.yuanyang.transform(s, a)
            s = s_
            step_num += 1
            if t:
                done = True

        return 1 if s == 9 else 0


if __name__ == "__main__":
    yuanyang = YuanYangEnv()
    brain = MC_RL(yuanyang)
    # 探索初始化方法
    qvalue2 = brain.mc_learning_ei(num_iter=10000)
    # on-policy方法
    # qvalue1=brain.mc_learning_on_policy(num_iter=10000, epsilon=0.2)
    print(qvalue2)
    # 将行为值函数渲染出来
    yuanyang.action_value = qvalue2
    ##########################################
    # 测试学到的策略
    flag = 1
    s = 0
    step_num = 0
    path = []
    # 将最优路径打印出来
    while flag:
        # 渲染路径点
        path.append(s)
        yuanyang.path = path
        print("Path updated to:", path)
        a = brain.greedy_policy(qvalue2, s)
        print('State:', s, 'Action:', a, 'Q-values:', qvalue2[s, :])
        yuanyang.bird_male_position = yuanyang.state_to_position(s)
        yuanyang.render()
        time.sleep(0.25)
        step_num += 1
        s_, r, t = yuanyang.transform(s, a)
        print('Next state:', s_, 'Reward:', r, 'Done:', t)
        if t or step_num > 30:
            flag = 0
        s = s_
    # 渲染最后的路径点
    yuanyang.bird_male_position = yuanyang.state_to_position(s)
    path.append(s)
    yuanyang.path = path
    yuanyang.render()
    while True:
        yuanyang.render()