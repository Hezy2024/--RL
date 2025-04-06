import numpy as np
import random
import os
import pygame
import time
import matplotlib.pyplot as plt
from yuanyang_env_td import *
from yuanyang_env_td import YuanYangEnv

class TD_RL:
    def __init__(self, yuanyang):
        self.gamma = yuanyang.gamma
        self.yuanyang = yuanyang
        #值函数的初始值
        self.qvalue=np.zeros((len(self.yuanyang.states),len(self.yuanyang.actions)))
    #定义贪婪策略
    def greedy_policy(self, qfun, state):
        amax=qfun[state,:].argmax()#找到动作值函数最大的动作
        return self.yuanyang.actions[amax]
    #定义epsilon贪婪策略
    def epsilon_greedy_policy(self, qfun, state, epsilon):
        amax = qfun[state, :].argmax()#找到动作值函数最大的动作
        # 概率部分
        if np.random.uniform() < 1 - epsilon:
            # 最优动作
            return self.yuanyang.actions[amax]
        else:
            return self.yuanyang.actions[int(random.random() * len(self.yuanyang.actions))]#随机选择一个动作
    #找到动作所对应的序号
    def find_anum(self,a):
        for i in range(len(self.yuanyang.actions)):
            if a==self.yuanyang.actions[i]:#找到动作的序号  
                return i

    def sarsa(self, num_iter, alpha, epsilon):
        for _ in range(num_iter):
            # 初始化状态
            s = 0
            # 根据epsilon-greedy策略选择动作
            a = self.epsilon_greedy_policy(self.qvalue, s, epsilon)
            done = False
            while not done:
                # 与环境交互
                s_next, r, done = self.yuanyang.transform(s, a)
                # 选择下一个动作
                a_next = self.epsilon_greedy_policy(self.qvalue, s_next, epsilon)#这是策略内的下一个动作，而不是环境的下一个动作，是on-policy的。
                # 获取动作序号
                a_num = self.find_anum(a)
                a_next_num = self.find_anum(a_next)
                # 更新Q值
                self.qvalue[s, a_num] = self.qvalue[s, a_num] + alpha * (r + self.gamma * self.qvalue[s_next, a_next_num] - self.qvalue[s, a_num])#这是TD的更新公式
                # Q(S_t, A_t) <- Q(S_t, A_t) + α * [R_{t+1} + γ * Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
                # 其中：
                # - Q(S_t, A_t) 是在状态 S_t 采取动作 A_t 时的Q值，对应代码中的 `self.qvalue[s, a_num]`
                # - α 是学习率，对应代码中的 `alpha`
                # - R_{t+1} 是在状态 S_t 采取动作 A_t 后获得的即时奖励，对应代码中的 `r`
                # - γ 是折扣因子，对应代码中的 `self.gamma`
                # - Q(S_{t+1}, A_{t+1}) 是在状态 S_{t+1} 采取动作 A_{t+1} 时的Q值，对应代码中的 `self.qvalue[s_next, a_next_num]`
                # 更新状态和动作
                s = s_next
                a = a_next

        return self.qvalue

    def qlearning(self, num_iter, alpha, epsilon):
        for _ in range(num_iter):
            # 初始化状态
            s = 0
            done = False
            while not done:
                # 根据epsilon-greedy策略选择动作
                a = self.epsilon_greedy_policy(self.qvalue, s, epsilon)
                # 与环境交互
                s_next, r, done = self.yuanyang.transform(s, a)
                # 获取动作序号
                a_num = self.find_anum(a)
                # 选择下一个状态的最优动作
                a_next_num = self.qvalue[s_next, :].argmax()#这是策略外的下一个动作，是off-policy的。
                # 更新Q值
                self.qvalue[s, a_num] = self.qvalue[s, a_num] + alpha * (r + self.gamma * self.qvalue[s_next, a_next_num] - self.qvalue[s, a_num])
                # 更新状态
                s = s_next

        return self.qvalue
    def greedy_test(self):
        s = 0
        s_sample = []
        done = False
        flag = 0
        step_num = 0
        while False == done and step_num < 30:
            a = self.greedy_policy(self.qvalue, s)
            # 与环境交互
            s_next, r, done = self.yuanyang.transform(s, a)
            s_sample.append(s)
            s = s_next
            step_num += 1
        if s == 9:
            flag = 1
        if s == 9 and step_num<21:
            flag = 2
        return flag

if __name__=="__main__":
    yuanyang = YuanYangEnv()
    brain = TD_RL(yuanyang)
    qvalue2 = brain.sarsa(num_iter =10000, alpha = 0.1, epsilon = 0.1)#这里一开始是epsilon=0.8，没注意到，导致一直没找到终点。
    #qvalue2=brain.qlearning(num_iter=10000, alpha=0.1, epsilon=0.1)
    #打印学到的值函数
    yuanyang.action_value = qvalue2
    ##########################################
    # 测试学到的策略
    flag = 1
    s = 0
    # print(policy_value.pi)
    step_num = 0
    path = []
    # 将最优路径打印出来
    while flag:
        # 渲染路径点
        path.append(s)
        yuanyang.path = path
        a = brain.greedy_policy(qvalue2, s)
        # a = agent.bolzman_policy(qvalue,s,0.1)
        print('%d->%s\t' % (s, a), qvalue2[s, 0], qvalue2[s, 1], qvalue2[s, 2], qvalue2[s, 3])
        yuanyang.bird_male_position = yuanyang.state_to_position(s)
        yuanyang.render()
        time.sleep(0.25)
        step_num += 1
        s_, r, t = yuanyang.transform(s, a)
        if t == True or step_num > 30:
            flag = 0
        s = s_
    # 渲染最后的路径点
    yuanyang.bird_male_position = yuanyang.state_to_position(s)
    path.append(s)
    yuanyang.render()
    while True:
        yuanyang.render()
#sarsa 算法是一种on-policy算法，它在更新Q值时使用当前策略选择的下一个动作；而 qlearning 算法是一种off-policy算法，它在更新Q值时使用下一个状态的最优动作。