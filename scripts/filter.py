# INSERT_YOUR_CODE
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# 假设某些函数用于生成/获取定位观测点
def get_observations():
    # 模拟实际数据，请替换为你的观测读取方式
    np.random.seed(42)
    true_pos = np.array([5.0, 5.0])
    pos_obs = []
    for _ in range(80):
        # 大部分点在±0.25m圈内
        obs = true_pos + np.random.normal(0, 0.18, size=2)
        pos_obs.append(obs)
    # 人为插入几个野点
    pos_obs[9] = true_pos + np.array([2.2, -1.7])
    pos_obs[32] = true_pos + np.array([0.9, 2.1])
    pos_obs[40] = true_pos + np.array([-1.5, -1.2])
    pos_obs = np.array(pos_obs)
    return pos_obs, true_pos

class SimpleKF2D:
    def __init__(self, process_var=0.05, measure_var=0.5):
        # x: [x, y], P: 协方差
        self.x = None
        self.P = None
        self.Q = np.eye(2) * process_var  # 保持0.05不变
        self.R = np.eye(2) * measure_var  # 可调
    def init(self, z_init):
        self.x = np.array(z_init)
        self.P = np.eye(2)
    def update(self, z):
        # 预测
        x_pred = self.x
        P_pred = self.P + self.Q
        # 更新
        K = P_pred @ np.linalg.inv(P_pred + self.R)
        self.x = x_pred + K @ (z - x_pred)
        self.P = (np.eye(2) - K) @ P_pred
        return self.x.copy()

# 融合和异常值剔除主流程
obs, true_pos = get_observations()
filtered = []
kf = SimpleKF2D(process_var=0.05, measure_var=0.5)  # measure_var=0.5 (变大)，process_var=0.05(不变)

THRESH = 0.6  # 剔除阈值收紧到0.6米

for idx, z in enumerate(obs):
    if idx == 0:
        kf.init(z)
        filtered.append(z)
        continue
    # 异常值检测
    pred = kf.x
    dist = np.linalg.norm(z - pred)
    if dist > THRESH:
        # print(f"第{idx}帧被丢弃, 偏差{dist:.2f}")
        filtered.append(pred)  # 维持原定状态，不输入该观测
        continue
    filtered.append(kf.update(z))

filtered = np.array(filtered)

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体（如有问题可选微软雅黑等）
matplotlib.rcParams['axes.unicode_minus'] = False    # 支持负号正常显示

# 重新画图，防止之前的内容影响
plt.figure(figsize=(7,7))
plt.grid(True)
plt.title("轨迹滤波-异常值剔除(阈值0.6米) & 滤波器更平滑")

# 原始观测点：橙色实心圆点，透明度更低
plt.scatter(obs[:,0], obs[:,1], color='orange', marker='o', s=38, alpha=0.33, label='原始定位')

# 滤波后轨迹：蓝色折线，并以空心圆显示滤波点
plt.plot(filtered[:,0], filtered[:,1], '-', color='dodgerblue', alpha=0.8, label='滤波后轨迹')
plt.scatter(filtered[:,0], filtered[:,1], facecolors='none', edgecolors='dodgerblue', marker='o', s=55, lw=1.2, alpha=0.85, label='滤波点')

# 红星: 真实位置
plt.scatter([true_pos[0]], [true_pos[1]], color='red', marker='*', s=200, label='真实位置')

# 绿色误差圈(0.6米)
circle = plt.Circle(true_pos, 0.6, color='limegreen', fill=False, linewidth=2, linestyle='--', label='0.6米误差圈')
plt.gca().add_patch(circle)

plt.xlabel('X (米)')
plt.ylabel('Y (米)')
plt.legend(loc='best')
plt.axis('equal')
plt.tight_layout()
plt.show()