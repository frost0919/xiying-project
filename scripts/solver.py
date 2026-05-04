import numpy as np


def tdoa_chan(anchors, time_diffs, speed_of_light=299792458):
    """
    基于Chan算法的TDOA二维定位函数

    参数说明:
    anchors : numpy.ndarray
        基站坐标，形状为(N, 2)，N为基站数，二维空间
    time_diffs : numpy.ndarray 或 list
        以第一个基站为参考的到达时间差(单位：秒)，长度为N-1
    speed_of_light : float
        光速，单位为m/s，默认299792458

    返回:
    tag_pos : numpy.ndarray
        标签的二维坐标, 形状为(2,)
    """
    if not isinstance(anchors, np.ndarray):
        anchors = np.array(anchors)
    if not isinstance(time_diffs, np.ndarray):
        time_diffs = np.array(time_diffs)

    N = anchors.shape[0]
    if N < 3:
        raise ValueError("至少需要3个基站进行TDOA定位")
    if anchors.shape[1] != 2:
        raise ValueError("anchors必须为二维坐标，形状为(N, 2)")
    if time_diffs.shape[0] != N - 1:
        raise ValueError("time_diffs长度必须为N-1")

    delta_r = time_diffs * speed_of_light

    x1, y1 = anchors[0]
    xi = anchors[1:, 0]
    yi = anchors[1:, 1]

    H = np.column_stack((xi - x1, yi - y1))
    ri2 = xi**2 + yi**2
    r12 = x1**2 + y1**2
    b = 0.5 * (ri2 - r12 - delta_r**2)

    try:
        pos, _, _, _ = np.linalg.lstsq(H, b, rcond=None)
    except np.linalg.LinAlgError as e:
        raise RuntimeError("线性方程组求解失败: " + str(e)) from e

    return pos
