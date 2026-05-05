# INSERT_YOUR_CODE

import numpy as np
import matplotlib.pyplot as plt
from solver import tdoa_chan
import matplotlib

def simulate_uwb_tdoa(
    anchors,
    true_pos,
    speed_of_light=299_792_458.0,
    nsigma=1e-9,
    n_trials=200
):
    """
    anchors: ndarray shape (3,2)
    true_pos: [x, y]
    speed_of_light: 光速 (m/s)
    nsigma: 时间噪声标准差 (秒)
    n_trials: 模拟次数
    """
    anchors = np.asarray(anchors)
    true_pos = np.asarray(true_pos)
    results = []

    for i in range(n_trials):
        # 1. 真实距离
        dists = np.linalg.norm(anchors - true_pos, axis=1)  # size 3

        # 2. 理论TOA
        toas = dists / speed_of_light  # 单位：秒

        # 3. 严格模拟TDOA (每个基站分别独立加噪声)
        noise = np.random.normal(0, nsigma, size=3)
        noisy_toas = toas + noise

        # 4. 基站间时间差 (以0号为参考)
        ref_toa = noisy_toas[0]
        t21 = noisy_toas[1] - ref_toa
        t31 = noisy_toas[2] - ref_toa
        time_diffs = np.array([t21, t31])

        # 5. Chan算法解算
        try:
            est_pos = tdoa_chan(anchors, time_diffs, speed_of_light)
            err = np.linalg.norm(est_pos - true_pos)
            results.append((est_pos, err))
        except Exception as e:
            print(f"Chan算法异常: {e}")
            continue

    est_positions = np.array([pos for pos, _ in results])
    errors = np.array([err for _, err in results])

    return est_positions, errors

if __name__ == "__main__":
    np.random.seed(42)
    # 1. 固定参数
    ANCHORS = np.array([[0,0], [10,0], [0,10]])
    TRUE_POS = np.array([5, 5])

    est_positions, errors = simulate_uwb_tdoa(
        ANCHORS,
        TRUE_POS,
        nsigma=1e-9,
        n_trials=200
    )

    print("平均定位误差(米): {:.4f}".format(errors.mean()))
    print("最大定位误差(米): {:.4f}".format(errors.max()))
    print("≤1米精度的比例: {:.1f}%".format(np.mean(errors < 1.0)*100))


    # 设置中文字体为黑体（防止出现乱码或方框）
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 也可以试 '微软雅黑' 等
    matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

    plt.figure(figsize=(6,6))
    plt.scatter(est_positions[:,0], est_positions[:,1], c='dodgerblue', alpha=0.7, label='Chan定位结果')
    plt.scatter([TRUE_POS[0]], [TRUE_POS[1]], c='red', marker='*', s=200, label='真实位置')
    plt.scatter(ANCHORS[:,0], ANCHORS[:,1], c='black', marker='^', s=80, label='基站')
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("3基站TDOA-UWB定位仿真 (Chan算法)")
    plt.legend()
    plt.grid(True)
    # 显示误差圈
    for pos in est_positions:
        plt.plot([TRUE_POS[0], pos[0]], [TRUE_POS[1], pos[1]], color='gray', alpha=0.1)
    plt.xlim(-2, 12)
    plt.ylim(-2, 12)
    plt.tight_layout()
    plt.show()