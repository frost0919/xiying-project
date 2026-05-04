import numpy as np

def tdoa_chan(anchors, time_diffs, speed_of_light=299792458):
    """
    基于Chan算法的TDOA二维定位
    :param anchors: 基站坐标（N,2），N>=3，numpy数组
    :param time_diffs: 以第一个基站为参考的到达时间差，长度为N-1，单位: 秒
    :param speed_of_light: 光速，默认299792458 m/s
    :return: 标签的二维坐标 (x, y)
    """
    # 错误处理
    if not isinstance(anchors, np.ndarray):
        raise TypeError("anchors必须为numpy数组")
    if anchors.ndim != 2 or anchors.shape[1] != 2:
        raise ValueError("anchors形状应为 (N,2)，每行一个基站的(x,y)坐标")
    N = anchors.shape[0]
    if N < 3:
        raise ValueError("至少需要3个基站进行二维定位")
    if len(time_diffs) != N - 1:
        raise ValueError("time_diffs长度应为N-1（以第1个基站为参考）")

    # 参考基站坐标(A1)
    x1, y1 = anchors[0]
    # 非参考基站坐标(A2...AN)
    anchors_rel = anchors[1:]

    # 将时间差转换为距离差 di1 = c * Δti1
    r = speed_of_light * np.array(time_diffs)  # 形状为(N-1,)

    # 计算每个基站到参考基站的(x, y)差值
    xi = anchors_rel[:, 0] - x1
    yi = anchors_rel[:, 1] - y1

    # 计算二次项 Si = xi^2 + yi^2
    Si = xi**2 + yi**2

    # 构建系数矩阵A和向量b
    # A * [x; y] = b
    # A的每一行为 [xi, yi]; b的每一项为 (Si - r_i^2)/2
    A = np.column_stack((xi, yi))
    b = 0.5 * (Si - r**2)

    # 用最小二乘法求解线性方程组 A * pos = b，得到参考点(0,0)下的解
    # 这一步为Chan算法的初始估计（LS解）
    try:
        pos_ls, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    except Exception as e:
        raise RuntimeError(f"最小二乘法求解失败: {e}")

    # 再将坐标平移回原始参考基站坐标系
    x, y = pos_ls + np.array([x1, y1])
    return np.array([x, y])
    # INSERT_YOUR_CODE
if __name__ == "__main__":
    # 直接运行可见输出结果
    def main():
        anchors = np.array([
            [0., 0.],
            [10., 0.],
            [0., 10.],
            [10., 10.]
        ])
        # 假设模拟的标签在 (3,4)，到各基站的真值距离
        pos_true = np.array([3, 4])
        c = 299792458
        dists = np.linalg.norm(anchors - pos_true, axis=1)
        timestamps = dists / c
        time_diffs = timestamps[1:] - timestamps[0]
        calc_pos = tdoa_chan(anchors, time_diffs)
        print(f"真坐标: ({pos_true[0]:.4f}, {pos_true[1]:.4f})")
        print(f"算法输出: ({calc_pos[0]:.4f}, {calc_pos[1]:.4f})")
        print(f"定位误差: {np.linalg.norm(calc_pos - pos_true):.6f} 米")

    main()
    # INSERT_YOUR_CODE
    # 让 if __name__ == '__main__': 块里的 main() 执行，无需手动调用
    # Python 文件被运行时，这块代码已经会被执行。
    # 你看到“依旧运行完没有结果”，大概率是因为直接“双击”或者用交互式环境运行脚本。
    # 请在终端里这样运行:
    #   python scripts/solver.py
    # 例如：
    #   cd 脚本文件所在目录（含有 scripts 文件夹）
    #   python scripts/solver.py
    # 如果依旧没输出，可以加一条明确输出，帮助确认 if __name__ == "__main__" 是否生效
    print("== Chan二维TDOA算法测试 ==")
    main()