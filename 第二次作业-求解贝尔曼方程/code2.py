import numpy as np

## 贝尔曼公式状态值求解
def closed_form_solution(R, P, gamma):
    # 获取行号
    n = R.shape[0]
    # 生成单位阵
    I = np.identity(n)
    
    # 计算 (I - gamma * P)
    M = I - gamma * P
    
    try:
        #计算矩阵的逆
        matrix_inverse = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        print(f"LinAlgError: Matrix is singular for gamma = {gamma}. Using pseudo-inverse.")
        matrix_inverse = np.linalg.pinv(M)  #使用伪逆

    # 计算状态价值
    V = matrix_inverse.dot(R)
    
    return V

def main():
    gammas = [0.5, 0.8, 1.0, 0.2]

    # 状态转移矩阵和奖励向量
    '''P = np.array([[0, 0.5, 0, 0, 0, 0.5,0],
                  [0, 0, 0.8, 0, 0, 0,0.2],
                  [0, 0, 0, 0.6, 0.4, 0,0],
                  [0, 0, 0, 0, 0, 0,1],
                  [0.2, 0.4, 0.4, 0, 0, 0,0],
                  [0.1, 0, 0, 0, 0, 0.9,0],
                  [0, 0, 0, 0, 0, 0,0]])
    
    R = np.array([-2, -2, -2, 10, -5, -1,0])   #c1,fb,c2,pub,c3,pass'''
    P = np.array([[0, 0.5, 0.5, 0, 0, 0,0],
                  [0.1, 0.9, 0, 0, 0, 0,0],
                  [0, 0, 0, 0, 0.8, 0,0.2],
                  [0.2, 0, 0.4, 0, 0.4, 0,0],
                  [0, 0, 0, 0.4, 0, 0.6,0],
                  [0, 0, 0, 0, 0, 0,1],
                  [0, 0, 0, 0, 0, 0,0]])
    
    R = np.array([-2, -1, -2, -5, -2, 10,0])  #c1,fb,c2,pub,c3,pass,sleep 

    # 计算每个折扣因子的状态价值
    for gamma in gammas:
        V = closed_form_solution(R, P, gamma)
        print(f"Values for gamma = {gamma}: {V}")

if __name__ == "__main__":
    main()
