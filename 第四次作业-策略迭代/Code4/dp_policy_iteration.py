import random
import time
from yuanyang_env import YuanYangEnv

class DP_Policy_Iter:
    def __init__(self, yuanyang):
        self.states = yuanyang.states
        self.actions = yuanyang.actions
        self.v = [0.0 for i in range(len(self.states)+1)]
        self.pi = dict()
        self.yuanyang = yuanyang
        self.gamma = yuanyang.gamma
        # 初始化策略
        for state in self.states:
            flag1 = 0
            flag2 = 0
            flag1 = yuanyang.collide(yuanyang.state_to_position(state))
            flag2 = yuanyang.find(yuanyang.state_to_position(state))
            if flag1 == 1 or flag2 == 1:
                continue
            self.pi[state] = self.actions[int(random.random() * len(self.actions))]
    
    def policy_evaluate(self):
        # 策略评估：计算值函数
        for i in range(10):
            delta = 0.0
            for state in self.states:
                flag1 = self.yuanyang.collide(self.yuanyang.state_to_position(state))
                flag2 = self.yuanyang.find(self.yuanyang.state_to_position(state))
                if flag1 == 1 or flag2 == 1:
                    continue
                a = self.pi[state]
                s_, r, t = self.yuanyang.transform(state, a)
                if t:
                    v_new = r
                else:
                    v_new = r + self.gamma * self.v[s_]
                delta += abs(self.v[state] - v_new)
                self.v[state] = v_new
            if delta < 1e-6:
                break
    
    def policy_improve(self):
        # 策略改进：利用更新后的值函数，进行策略改进（贪心策略）
        for state in self.states:
            flag1 = self.yuanyang.collide(self.yuanyang.state_to_position(state))
            flag2 = self.yuanyang.find(self.yuanyang.state_to_position(state))
            if flag1 == 1 or flag2 == 1:
                continue
            action_values = []
            for action in self.actions:
                s_, r, t = self.yuanyang.transform(state, action)
                if t:
                    action_value = r
                else:
                    action_value = r + self.gamma * self.v[s_]
                action_values.append(action_value)
            max_value = max(action_values)
            #找到所有动作中价值最大的动作
            best_actions = [self.actions[i] for i, value in enumerate(action_values) if value == max_value]
            #随机选择一个最优动作
            self.pi[state] = random.choice(best_actions)
    
    def policy_iterate(self):
        for i in range(100): #至少9轮才能覆盖全局，确保从起始点到终点的路径是准确的
            # 策略评估，更新值函数v
            self.policy_evaluate()
            # 策略改进，更新策略pi
            pi_old = self.pi.copy()
            self.policy_improve()
            if self.pi == pi_old:
                print("策略改善次数", i)
                break

if __name__ == "__main__":
    yuanyang = YuanYangEnv()
    policy_value = DP_Policy_Iter(yuanyang)
    policy_value.policy_iterate()
    flag = 1
    s = 10
    path = []
    # 将v值打印出来
    for state in range(100):
        i = int(state / 10)
        j = state % 10
        yuanyang.value[j, i] = policy_value.v[state]
    step_num = 0
    # 将最优路径打印出来
    while flag:
        # 渲染路径点
        path.append(s)
        yuanyang.path = path
        a = policy_value.pi[s]
        print('%d->%s\t' % (s, a))
        yuanyang.bird_male_position = yuanyang.state_to_position(s)
        yuanyang.render()
        time.sleep(0.2)
        step_num += 1
        s_, r, t = yuanyang.transform(s, a)
        if t == True or step_num > 200:
            flag = 0
        s = s_
    # 渲染最后的路径点
    yuanyang.bird_male_position = yuanyang.state_to_position(s)
    path.append(s)
    yuanyang.render()
    while True:
        yuanyang.render()
