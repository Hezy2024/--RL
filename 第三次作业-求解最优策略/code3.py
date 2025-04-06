import random

class Env(object):
    def __init__(self):
        self.S = ['s1', 's2', 's3', 's4', 's5']  # 状态集合

    def step(self, s, a):  # 状态转移函数和奖励函数
        s_n = None
        r = None
        terminal = False   # 是否进入终止状态
        # 实现自己的代码  附上状态转移过程和奖励值
        if s == 's1':
            if a == 'quit':
                r = 0
                s_n = 's2'  
            elif a == 'phone':
                r = -1
                s_n = 's1'
        
        elif s == 's2':
            if a == 'phone':
                r = -1
                s_n = 's1'
            elif a == 'study':
                r = -2
                s_n = 's3'  
        
        elif s == 's3':
            if a == 'study':
                r = -2
                s_n = 's4'
            elif a == 'sleep':
                r = 0
                s_n = 's5'  # 终止状态
                terminal = True
        
        elif s == 's4':
            if a == 'review':
                r = 10
                s_n = 's5'  # 终止状态
                terminal = True
            elif a == 'noreview':
                r = -5
                 # 根据概率转移到 s2, s3, s4
                rand=random.random()
                if rand < 0.2:
                    s_n = 's2'
                elif 0.2<= rand < 0.6:  # 0.2 + 0.4
                    s_n = 's3'
                else:
                    s_n = 's4'
                
        elif s == 's5':
            terminal = True  # 已经是终止状态

        return s_n, r, terminal


class Agent(object):
    def __init__(self):
        self.A = ['quit', 'phone', 'study', 'sleep', 'review', 'noreview']
        self.available_actions = {
            's1': ['phone', 'quit'],
            's2': ['phone', 'study'],
            's3': ['study', 'sleep'],
            's4': ['review', 'noreview']
        }

    def random_policy(self, s):
        a = None
        # 实现自己的代码  根据状态做到随机
        a = random.choice(self.available_actions[s])  #从可用动作中随机选择
        return a


if __name__ == "__main__":
    # 仿真随机策略
    # 寻找最优策略
    # 我这里给出一次仿真的示例, 假设初始状态是s2
    env = Env()
    agent = Agent()
    gamma =1
    #gamma = 1
    max_time_step = 1000
    s = 's2'
    curr_gamma = 1
    g = 0  # 这次实验的回报值, 多次实验后平均，即得到v(s2)的估计
    for i in range(max_time_step):
        a = agent.random_policy(s)
        s, r, term = env.step(s,a)
        
        g += curr_gamma * r
        curr_gamma *= gamma
        if term:
            break
    print(f"Estimated return for state s2: {g}")
    #Estimated return for state s2: -1.341796875
    #Estimated return for state s2: 6