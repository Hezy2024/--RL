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
        amax=qfun[state,:].argmax()
        return self.yuanyang.actions[amax]
    #定义epsilon贪婪策略
    def epsilon_greedy_policy(self, qfun, state, epsilon):
        amax = qfun[state, :].argmax()
        # 概率部分
        if np.random.uniform() < 1 - epsilon:
            # 最优动作
            return self.yuanyang.actions[amax]
        else:
            return self.yuanyang.actions[int(random.random() * len(self.yuanyang.actions))]
    #找到动作所对应的序号
    def find_anum(self,a):
        for i in range(len(self.yuanyang.actions)):
            if a==self.yuanyang.actions[i]:
                return i

    def qlearning(self,num_iter, alpha, epsilon):
        iter_num = []
        self.qvalue = np.zeros((len(self.yuanyang.states), len(self.yuanyang.actions)))
        #大循环
        for iter in range(num_iter):
            #随机初始化状态
            s = self.yuanyang.reset()
            # 随机选初始动作
            a = self.epsilon_greedy_policy(self.qvalue, s, epsilon)
            flag = self.greedy_test()
            if flag == 1:
                iter_num.append(iter)
                if len(iter_num)<2:#第一次完成任务
                    print("qlearning 第一次完成任务需要的迭代次数为：", iter_num[0])
            if flag == 2:
                print("qlearning 第一次实现最短路径需要的迭代次数为：", iter)
                break
            s_sample = []

            while True:
                # 与环境交互得到下一个状态
                s_, r, t = self.yuanyang.transform(s, a)
                # 利用td方法更新动作值函数
                a_ = self.greedy_policy(self.qvalue, s_)
                a_num = self.find_anum(a)
                self.qvalue[s, a_num] = self.qvalue[s, a_num] + alpha * (r + self.gamma * self.qvalue[s_, self.find_anum(a_)] - self.qvalue[s, a_num])
                # 行为策略
                a = self.epsilon_greedy_policy(self.qvalue, s_, epsilon)
                s = s_
                if t:
                    break
                # print(r)
        return self.qvalue

    def greedy_test(self):
        s = 0
        step_num = 0
        while True:
            a = self.greedy_policy(self.qvalue, s)
            s_, r, t = self.yuanyang.transform(s, a)
            step_num += 1
            s = s_
            if s == 9:
                flag = 1
                if step_num < 21:
                    flag = 2
                return flag
            if t or step_num > 30:
                return 0

if __name__=="__main__":
    yuanyang = YuanYangEnv()
    brain = TD_RL(yuanyang)
    #qvalue1 = brain.sarsa(num_iter =5000, alpha = 0.1, epsilon = 0.8)
    qvalue2=brain.qlearning(num_iter=10000, alpha=0.1, epsilon=0.1)
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