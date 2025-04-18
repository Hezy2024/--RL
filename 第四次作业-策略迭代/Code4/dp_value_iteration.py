import random
import time
from yuanyang_env import YuanYangEnv
import os

class DP_Value_Iter:
    def __init__(self, yuanyang):
        self.states = yuanyang.states
        self.actions = yuanyang.actions
        self.v = [0.0 for _ in range(len(self.states))]
        self.pi = dict()
        self.yuanyang = yuanyang
        self.gamma = yuanyang.gamma
        
        for state in self.states:
            flag1 = yuanyang.collide(yuanyang.state_to_position(state))
            flag2 = yuanyang.find(yuanyang.state_to_position(state))
            if flag1 == 1 or flag2 == 1:
                continue
            self.pi[state] = random.choice(self.actions)

    def value_iteration(self, theta=1e-6):
        while True:
            delta = 0
            for state in self.states:
                position = self.yuanyang.state_to_position(state)
                if self.yuanyang.collide(position) or self.yuanyang.find(position):
                    continue
                
                v_old = self.v[state]
                action_values = []
                
                for action in self.actions:
                    next_state, reward, terminal = self.yuanyang.transform(state, action)
                    if terminal:
                        action_value = reward
                    else:
                        action_value = reward + self.gamma * self.v[next_state]
                    action_values.append(action_value)

                self.v[state] = max(action_values)
                delta = max(delta, abs(v_old - self.v[state]))

            if delta < theta:
                break

        # 策略提取
        for state in self.states:
            position = self.yuanyang.state_to_position(state)
            if self.yuanyang.collide(position) or self.yuanyang.find(position):
                continue

            action_values = []
            for action in self.actions:
                next_state, reward, terminal = self.yuanyang.transform(state, action)
                if terminal:
                    action_value = reward
                else:
                    action_value = reward + self.gamma * self.v[next_state]
                action_values.append(action_value)

            best_action = self.actions[action_values.index(max(action_values))]
            self.pi[state] = best_action

if __name__ == "__main__":
    yuanyang = YuanYangEnv()
    policy_value = DP_Value_Iter(yuanyang)
    policy_value.value_iteration()
    
    # 将v值打印出来
    s = 0
    path = []
    for state in range(100):
        i = int(state / 10)
        j = state % 10
        yuanyang.value[j, i] = policy_value.v[state]
    flag = 1
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
        if t == True or step_num > 20:
            flag = 0
        s = s_

    # 渲染最后的路径点
    yuanyang.bird_male_position = yuanyang.state_to_position(s)
    path.append(s)
    yuanyang.render()
    while True:
        yuanyang.render()

        
