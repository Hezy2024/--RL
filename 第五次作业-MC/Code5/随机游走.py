import random
import numpy as np
import matplotlib.pyplot as plt

class ENV:
    def __init__(self):
        self.states = ['LT', 'A', 'B', 'C', 'D', 'E', 'RT']

    def reset(self):
        return self.states[3]

    def step(self, s, a):
        terminal = False  # 是否进入终止状态
        #s_n
        if a == 0: # a==0表示向左，a==1表示向右
            s_n = self.states[s - 1]
        else:
            s_n = self.states[s + 1]

        #r
        if s_n == 'RT':
            r = 1
        else:
            r = 0

        #terminal
        if s_n == "LT" or s_n == 'RT':
            terminal = True
        return s_n, r, terminal

class Methods:
    def __init__(self, env):
        self.actions = ['left', 'right'] #0为左 1为右
        self.gamma = gamma
        self.value = np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 1]) # 初始价值函数
        self.value_true = np.array([0,1/6, 2/6, 3/6, 4/6, 5/6, 1])
        self.env = env

    # 定义随机策略
    def random_policy(self):
        a_probs = np.array([0.5, 0.5])
        a = np.random.choice(self.actions, p=a_probs)
        return a

    #找到状态对应的序号
    def find_snum(self, s):
        for i in range(len(self.env.states)):
            if s == self.env.states[i]:
                return i

    #找到动作所对应的序号
    def find_anum(self, a):
        for i in range(len(self.actions)):
            if a == self.actions[i]:
                return i

    def TD_zero(self, alpha):#用TD，一步一步的更新状态
        #产生一幕的序列 在函数外调用产生一定数量的序列
        # 采集状态样本
        s_sample = []
        # 初始化状态 C
        s = self.env.reset()
        s_num = self.find_snum(s)
        a = self.random_policy()
        a_num = self.find_anum(a)
        done = False

        while done == False:
            # 与环境交互
            s_n, r, done = env.step(s_num, a_num)
            s_n_num = self.find_snum(s_n)

            # 存储数据，采样数据
            s_sample.append(s)

            # 更新值函数
            if s_n == 'RT':
                self.value[s_num] += alpha * (r - self.value[s_num])
            else:
                self.value[s_num] += alpha * (r + self.value[s_n_num] - self.value[s_num])
            # 转移到下一个状态，继续实验，s0-s1-s2
            s = s_n
            a = self.random_policy()
            s_num = self.find_snum(s)
            a_num = self.find_anum(a)

        return self.value

    def MC(self, alpha):
        # 采集状态样本
        s_sample = []
        # reward列表
        r_sample = []
        # 初始化状态 C
        s = self.env.reset()
        s_num = self.find_snum(s)
        a = self.random_policy()
        a_num = self.find_anum(a)
        done = False
        # 生成一幕序列
        while done == False:
            # 与环境交互
            s_n, r, done = env.step(s_num, a_num)
            s_n_num = self.find_snum(s_n)

            # 存储数据，采样数据
            s_sample.append(s)
            r_sample.append(r)

            # 转移到下一个状态，继续实验，s0-s1-s2
            s = s_n
            a = self.random_policy()
            s_num = self.find_snum(s)
            a_num = self.find_anum(a)

        #print(s_sample)
        #print(r_sample)
        # 更新价值
        for i in range(len(s_sample)):
            # 每次访问型 MC更新 因为只有最右停止reward是1 其他是0 且折扣为1 所以Gt直接用sum(reward)来求
            s_num = self.find_snum(s_sample[i])
            self.value[s_num] += alpha * (sum(r_sample) - self.value[s_num])#用整个策略的回报更新状态
        #print(self.value)
        return self.value

    def RMS_error(self, values):
        rmse = np.sqrt(np.sum(np.power(self.value_true[1:6] - values[1:6], 2)) / 5.0)
        return rmse

    def plot_value(self):
        plt.plot(self.value_true, label='value_true')
        for iter in [0, 1, 10, 300,500]:
            # 使用前需要重置value 否则会在原基础上进行叠加
            self.value = np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 1])
            for j in range(iter+1): #episodes
                td_value = self.TD_zero(0.1)
            plt.plot(td_value, label=str(iter) + ' episodes' + 'value')
        plt.xticks([0, 1, 2, 3, 4, 5, 6], env.states)
        plt.legend()
        plt.show()

    def plot_alpha(self, method, N, episodes):
        for alpha in [0.15, 0.1, 0.05]:
            total_error = np.zeros(episodes + 1)
            for run in range(N):
                rms_error = []
                #使用前需要重置value 否则会在原基础上进行叠加
                self.value = np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 1])
                for i in range(episodes + 1):
                    if method == 'TD':
                        values = self.TD_zero(alpha)
                    else:
                        values = self.MC(alpha)
                    rmse = self.RMS_error(values)
                    rms_error.append(rmse)
                total_error += np.asarray(rms_error)
            total_error /= N
            plt.plot(total_error, label='alpha=' + str(alpha))
        plt.legend()
        plt.show()

    def compare_2method(self, N, episodes):
        for alpha in [0.15, 0.1, 0.05]:
            total_error1 = np.zeros(episodes + 1)
            total_error2 = np.zeros(episodes + 1)
            for run in range(N):
                rms_error1 = []
                rms_error2 = []
                #使用前需要重置value 否则会在原基础上进行叠加
                self.value = np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 1])
                for i in range(episodes + 1):
                    values1 = self.TD_zero(alpha)
                    rmse1 = self.RMS_error(values1)
                    rms_error1.append(rmse1)
                self.value = np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 1])
                for i in range(episodes + 1):
                    values2 = self.MC(alpha)
                    rmse2 = self.RMS_error(values2)
                    rms_error2.append(rmse2)

                total_error1 += np.asarray(rms_error1)
                total_error2 += np.asarray(rms_error2)
            total_error1 /= N
            plt.plot(total_error1, label='TD ' + 'alpha=' + str(alpha))
            total_error2 /= N
            plt.plot(total_error2, label='MC ' + 'alpha=' + str(alpha))
        plt.legend()
        plt.show()

if __name__=="__main__":
    env = ENV()
    gamma = 1
    method = Methods(env)
    # #绘制alpha=0.1时，TD的价值函数
    method.plot_value()
    # #绘制不同alpha下的TD的RMS误差
    method.plot_alpha('TD', 100, 100)
    # #绘制不同alpha下的MC的RMS误差
    method.plot_alpha('MC',100, 100)
    #比较两者的收敛速度
    method.compare_2method(100, 100)




