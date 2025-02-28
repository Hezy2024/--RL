import numpy as np

def value_iteration(P, R, gamma, theta=1e-10):
    # 获取状态数
    n = R.shape[0]
    # 初始化状态价值向量
    V = np.zeros(n)
    
    while True:
        delta = 0  # 用于跟踪最大变化
        # 计算每个状态的价值
        for s in range(n):
            v = V[s]  # 保存当前的价值
            # 计算新价值
            V[s] = sum(P[s, a] * (R[s] + gamma * V[a]) for a in range(n))
            delta = max(delta, abs(v - V[s]))  # 更新最大变化
        # 如果变化小于阈值，停止迭代
        if delta < theta:
            break
            
    return V

def main():
    gammas = [0.5, 0.8, 1.0, 0.2]

    # 状态转移矩阵和奖励向量
    P = np.array([[0, 0.5, 0.5, 0, 0, 0,0],
                  [0.1, 0.9, 0, 0, 0, 0,0],
                  [0, 0, 0, 0, 0.8, 0,0.2],
                  [0.2, 0, 0.4, 0, 0.4, 0,0],
                  [0, 0, 0, 0, 0.4, 0.6,0],
                  [0, 0, 0, 0, 0, 0,1],
                  [0, 0, 0, 0, 0, 0,0]])
    
    R = np.array([-2, -1, -2, -5, -2, 10,0])  # 奖励向量

    # 计算每个折扣因子的状态价值
    for gamma in gammas:
        V = value_iteration(P, R, gamma)
        print(f"Values for gamma = {gamma}: {V}")

if __name__ == "__main__":
    main()