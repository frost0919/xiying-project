import numpy as np
import matplotlib.pyplot as plt

from solver import tdoa_chan


    # 只用三个基站的仿真
def main_three_anchors():
    # 固定三个基站坐标
    anchors = np.array([
        [0., 0.],
        [10., 0.],
        [0., 10.]
    ])
    # 随机生成标签的真实坐标
    np.random.seed(42)  # 保证可重复
    real_pos = np.random.uniform(low=0, high=10, size=2)
    print(f"随机生成标签的真实坐标: {real_pos}")

    # 计算标签到各基站的真实距离
    dists = np.linalg.norm(anchors - real_pos, axis=1)

    # 理想无噪声到达时间戳
    c = 299_792_458  # 光速，米/秒
    time_stamps = dists / c

    # 加入5ns高斯噪声
    noise_ns = np.random.normal(loc=0.0, scale=5, size=anchors.shape[0])
    noise_s = noise_ns * 1e-9
    noisy_time_stamps = time_stamps + noise_s

    # 以第一个基站为参考
    ref_time = noisy_time_stamps[0]
    time_diffs = noisy_time_stamps[1:] - ref_time

    # 调用Chan算法
    calc_pos = tdoa_chan(anchors, time_diffs, speed_of_light=c)

    # 定位误差
    error = np.linalg.norm(calc_pos - real_pos)

    print(f"[三基站仿真] 真实坐标:  ({real_pos[0]:.4f}, {real_pos[1]:.4f})")
    print(f"[三基站仿真] 计算坐标: ({calc_pos[0]:.4f}, {calc_pos[1]:.4f})")
    print(f"[三基站仿真] 定位误差: {error:.4f} 米")

    # 绘图
    plt.figure(figsize=(6, 6))
    plt.scatter(anchors[:, 0], anchors[:, 1], c='b', marker='s', s=100, label="基站")
    plt.scatter(real_pos[0], real_pos[1], c='g', marker='o', s=100, label="真实标签")
    plt.scatter(calc_pos[0], calc_pos[1], c='r', marker='x', s=100, label="计算标签")
    for i, (ax, ay) in enumerate(anchors):
        plt.text(ax, ay-0.4, f"A{i+1}", ha='center', va='top', fontsize=10, color='b')
    plt.legend()
    plt.title("TDOA Chan算法定位-三基站仿真")
    plt.xlim(-1, 11)
    plt.ylim(-1, 11)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# 运行三基站仿真
main_three_anchors()