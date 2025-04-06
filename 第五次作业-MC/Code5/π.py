import random

def monte_carlo_simulation(n):
    points_in_shadow = 0
    points_in_quarter_circle = 0

    for _ in range(n):
        x = random.uniform(0, 2)
        y = random.uniform(0, 2)

        # 判断是否在四分之一大圆内（用于计算π）
        if x**2 + y**2 <= 4:
            points_in_quarter_circle += 1
        
        # 判断是否在小圆内且在大圆外（阴影区域）
        if (x-1)**2 + (y-1)**2 <= 1 and x**2 + y**2 > 4:
            points_in_shadow += 1

    # 计算结果
    pi = (points_in_quarter_circle / n) * 4  # 四分之一圆面积=π
    shadow_area = (points_in_shadow / n) * 4  # 阴影区域面积
    return pi, shadow_area

# 执行计算
num_points = int(input("请输入随机点数（如1,000,000）: "))
pi, shadow = monte_carlo_simulation(num_points)

# 输出结果
print(f"估算π值: {pi:.6f} (理论π: 3.1415926)")
print(f"阴影面积: {shadow:.6f}")