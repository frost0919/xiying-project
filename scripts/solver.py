# INSERT_YOUR_CODE

import numpy as np

def tdoa_chan(anchors, time_diffs, speed_of_light):
    """
    基于Chan算法的3基站TDOA二维定位
    anchors: ndarray, shape (3, 2), 3个基站坐标, 格式如 [[x1, y1], [x2, y2], [x3, y3]]
    time_diffs: ndarray, shape (2,), 每个为 anchor2-ref, anchor3-ref 的时间差 (单位:s)，其中ref为anchors[0]
    speed_of_light: float, 光速
    返回: 标签二维坐标 [x, y]
    """

    anchors = np.asarray(anchors)
    if anchors.shape != (3,2) or len(time_diffs)!=2:
        raise ValueError("anchors需为(3,2)，time_diffs需为2元素")
    # 变量名约定:
    # ref为anchors[0], i=1指anchors[1], i=2指anchors[2]
    
    x1, y1 = anchors[0]
    x2, y2 = anchors[1]
    x3, y3 = anchors[2]
    # T_10 = 到达anchor2-到达anchor1; T_20 = 到达anchor3-到达anchor1
    t21, t31 = time_diffs  # t21 = t2-t1, t31 = t3-t1
    c = speed_of_light

    # 步骤1: 构建加权最小二乘的线性方程 Ax=b, 解得初值
    # 定义
    # Ri = sqrt((x-xi)^2+(y-yi)^2)
    # τ21 = t21 = (R2-R1)/c, τ31 = t31 = (R3-R1)/c

    # 线性化处理得到（公式见Chan1987论文 两步法的第一步）:

    # 构造
    # h = [x2^2 + y2^2 - x1^2 - y1^2 - c^2*t21^2
    #      x3^2 + y3^2 - x1^2 - y1^2 - c^2*t31^2 ]/2
    h = np.array([
        (x2**2 + y2**2 - x1**2 - y1**2 - c**2 * t21**2)/2,
        (x3**2 + y3**2 - x1**2 - y1**2 - c**2 * t31**2)/2
    ])
    # G = [ [x2-x1, y2-y1], [x3-x1, y3-y1] ]
    G = np.array([
        [x2-x1, y2-y1],
        [x3-x1, y3-y1]
    ])
    # q=[c*t21, c*t31]
    q = np.array([c*t21, c*t31])
    Q = np.diag(q**2)  # 观测噪声协方差的近似，加权时使用
    # W = Q^-1
    try:
        W = np.linalg.inv(Q)
    except np.linalg.LinAlgError:
        W = np.eye(2)

    # 初始线性解 (公式11)
    GTWG = G.T @ W @ G
    GTWh = G.T @ W @ h
    try:
        pos_0 = np.linalg.solve(GTWG, GTWh)
    except np.linalg.LinAlgError:
        # 回退成未加权
        pos_0 = np.linalg.lstsq(G, h, rcond=None)[0]
    
    # 步骤2: 非线性校正——一次Chan算法的“修正”(见Chan1987，式13~16)
    # 利用初始结果，计算Ri0
    x0, y0 = pos_0
    R10 = np.sqrt((x0-x1)**2 + (y0-y1)**2)
    R20 = np.sqrt((x0-x2)**2 + (y0-y2)**2)
    R30 = np.sqrt((x0-x3)**2 + (y0-y3)**2)

    # D21 = (x2-x1)*(x0-x1) + (y2-y1)*(y0-y1)
    # D31 = (x3-x1)*(x0-x1) + (y3-y1)*(y0-y1)
    D21 = (x2-x1)*(x0-x1) + (y2-y1)*(y0-y1)
    D31 = (x3-x1)*(x0-x1) + (y3-y1)*(y0-y1)
    # 构造修正的G’
    Gp = np.array([
        [(x2-x1)/R10 - (x0-x1)*(D21/R10**3), (y2-y1)/R10 - (y0-y1)*(D21/R10**3)],
        [(x3-x1)/R10 - (x0-x1)*(D31/R10**3), (y3-y1)/R10 - (y0-y1)*(D31/R10**3)],
    ])
    # 修正右端项
    hp = np.array([
        c*t21 - (R20-R10),
        c*t31 - (R30-R10)
    ])

    # 再加权最小二乘 (式16)
    try:
        d_theta = np.linalg.solve(Gp.T @ Gp, Gp.T @ hp)
    except np.linalg.LinAlgError:
        d_theta = np.linalg.lstsq(Gp, hp, rcond=None)[0]
    pos_final = pos_0 + d_theta
    return pos_final

# ——测试用例——
def test_chan_3anchors():
    """
    测试3基站Chan算法是否能有效降到1米以内
    """
    np.random.seed(0)
    anchors = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [0.0, 10.0]
    ])
    true_pos = np.array([5.0, 5.0])
    c = 299_792_458.0 # m/s

    # 真实距离
    r = np.linalg.norm(anchors - true_pos, axis=1)  # r1, r2, r3

    # 理想TDOA: t21 = (r2-r1)/c, t31 = (r3-r1)/c
    t21 = (r[1] - r[0]) / c
    t31 = (r[2] - r[0]) / c
    # 加入±1ns高斯噪声 (实际sigma=1ns，即1e-9s)
    noise = np.random.normal(0, 1e-9, 2)
    td_noisy = np.array([t21, t31]) + noise

    pos = tdoa_chan(anchors, td_noisy, speed_of_light=c)

    err = np.linalg.norm(pos - true_pos)

    print("基站坐标:", anchors)
    print("真实标签坐标:", true_pos)
    print("带噪声的TDOA[ns]:", td_noisy*1e9)
    print("Chan算法输出坐标: [{:.4f}, {:.4f}]".format(*pos))
    print("定位误差: {:.2f} 米".format(err))
    assert err < 1.0, f"误差大于1米! ({err}m)"

if __name__ == "__main__":
    test_chan_3anchors()