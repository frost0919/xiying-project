# INSERT_YOUR_CODE
import sys
import time
import csv
import os

# 串口部分
try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("未安装 pyserial 库，请先运行 pip install pyserial")
    sys.exit(1)

# 算法部分
try:
    from solver import tdoa_chan as tdoa_chan_solver
except ImportError:
    print("警告: 未找到 solver.py 或 tdoa_chan_solver 函数，相关功能不可用")
    tdoa_chan_solver = None

try:
    from filter import SimpleKF2D as KalmanFilter2D
except ImportError:
    print("警告: 未找到 filter.py 或 SimpleKF2D（KalmanFilter2D）类，相关功能不可用")
    KalmanFilter2D = None

# ==== 常量配置部分（根据实际调整） ====
ANCHORS = [
    [0.0, 0.0],
    [10.0, 0.0],
    [0.0, 10.0]
]
SPEED_OF_LIGHT = 299_792_458.0  # m/s
THRESH = 0.6  # 异常值剔除阈值（米）
BAUD_DEFAULT = 115200
CSV_FILE = "uwb_data.csv"
ENABLE_FILTER = True  # 开关: True-输出滤波, False-只输出原始定位

# ======= 串口工具函数 =======

def list_ports():
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]

def select_serial_port():
    ports = list_ports()
    if not ports:
        print("未检测到可用串口。")
        return None
    print("可用串口列表：")
    for idx, port in enumerate(ports):
        print(f"  {idx+1}: {port}")
    while True:
        sel = input("请选择串口号 (输入编号 or 直接输入端口名，如 COM3 或 /dev/ttyUSB0): ").strip()
        if sel.isdigit():
            i = int(sel) - 1
            if 0 <= i < len(ports):
                return ports[i]
            else:
                print("无效编号，请重试。")
        elif sel in ports:
            return sel
        else:
            print("未识别的输入，请重试。")

# ======= 串口数据解析 =======
# 假设协议: 收到一帧包含3个基站的时间戳(单位:ns)，格式如: "TS:12345,23456,34567\n"
# 可以根据实际串口数据协议调整
def parse_uwb_frame(line):
    """
    解析一帧串口数据，返回: (t1, t2, t3) 单位:秒, 或 None
    """
    try:
        # 支持格式: "TS:123456,234567,345678"
        if isinstance(line, bytes):
            try:
                line = line.decode('utf-8')
            except Exception:
                line = line.decode('gbk', errors='ignore')
        line = line.strip()
        if not line or not line.startswith("TS:"):
            return None
        ts_str = line[3:].split(",")
        if len(ts_str) != 3:
            return None
        ts_ns = [int(x.strip()) for x in ts_str]
        ts_sec = [x * 1e-9 for x in ts_ns]  # 转为秒
        return tuple(ts_sec)
    except Exception:
        return None

# ======= 实时主循环 =======
def main():
    print("==== UWB 串口定位模块 ====")
    port = select_serial_port()
    if not port:
        print("未选择串口，退出。")
        return
    try:
        baud = int(input(f"输入波特率 (回车默认 {BAUD_DEFAULT}): ").strip() or BAUD_DEFAULT)
    except ValueError:
        baud = BAUD_DEFAULT

    print(f"打开串口：{port}，波特率：{baud}")
    try:
        ser = serial.Serial(port, baudrate=baud, timeout=1)
    except Exception as e:
        print(f"打开串口失败: {e}")
        return

    if not tdoa_chan_solver or (ENABLE_FILTER and not KalmanFilter2D):
        print("算法/滤波依赖未就绪，无法正常运行。")
        ser.close()
        return

    # Kalman Filter初始化
    if ENABLE_FILTER:
        kf = KalmanFilter2D(process_var=0.05, measure_var=0.5)
        kf_inited = False

    # 写入CSV准备
    csv_header = ["recv_time", "t1_ns", "t2_ns", "t3_ns", "x_raw", "y_raw", "x_filt", "y_filt"]
    file_exists = os.path.isfile(CSV_FILE)
    csv_f = open(CSV_FILE, mode="a", newline="")
    csv_writer = csv.writer(csv_f)
    if not file_exists or os.stat(CSV_FILE).st_size == 0:
        csv_writer.writerow(csv_header)
        csv_f.flush()

    print(f"开始监听串口数据…… (保存至 {CSV_FILE})")
    print("按 Ctrl+C 退出。\n")

    try:
        while True:
            try:
                line = ser.readline()
                if not line:
                    continue
                ts = parse_uwb_frame(line)
                # 打印原始帧内容调试
                # print(f"串口原始: {repr(line)}")
                if ts is None:
                    continue
                t1, t2, t3 = ts
                time_diffs = [t2-t1, t3-t1]  # 以 t1 为参考 (顺序: t2-t1, t3-t1)
                try:
                    est_pos = tdoa_chan_solver(ANCHORS, time_diffs, SPEED_OF_LIGHT)
                except Exception as ex:
                    print(f"定位解算异常: {ex}")
                    continue
                x_raw, y_raw = est_pos
                # 滤波
                if ENABLE_FILTER:
                    if not kf_inited:
                        kf.init([x_raw, y_raw])
                        x_filt, y_filt = x_raw, y_raw
                        kf_inited = True
                    else:
                        pred = kf.x
                        dist = ((x_raw - pred[0])**2 + (y_raw - pred[1])**2)**0.5
                        if dist > THRESH:
                            print(f"[异常值剔除] {time.strftime('%H:%M:%S')}，原始:({x_raw:.3f},{y_raw:.3f}) 偏离滤波器{dist:.2f}米 —— 丢弃")
                            x_filt, y_filt = pred[0], pred[1]
                        else:
                            x_filt, y_filt = kf.update([x_raw, y_raw])
                else:
                    x_filt, y_filt = '', ''
                # 打印到终端
                print(f"[{time.strftime('%H:%M:%S')}] TS(ns):{int(t1*1e9)},{int(t2*1e9)},{int(t3*1e9)} | "
                      f"原始坐标:[{x_raw:.3f},{y_raw:.3f}]", end='')
                if ENABLE_FILTER:
                    print(f" | 滤波:[{x_filt:.3f},{y_filt:.3f}]")
                else:
                    print()
                # 记录到CSV
                row = [time.time(), int(t1*1e9), int(t2*1e9), int(t3*1e9), round(x_raw, 6), round(y_raw, 6)]
                if ENABLE_FILTER:
                    row += [round(x_filt, 6), round(y_filt, 6)]
                else:
                    row += ['', '']
                csv_writer.writerow(row)
                csv_f.flush()
            except KeyboardInterrupt:
                print("用户中断，退出。")
                break
            except Exception as e:
                print(f"运行时异常: {e}. 5秒后重试。")
                time.sleep(5)
                continue
    finally:
        ser.close()
        csv_f.close()
        print("串口关闭，文件保存。")

# ========== 离线画图对比 ==========
def plot_trajectory_from_csv(csv_file=CSV_FILE):
    """
    读取 csv 文件，画原始点+滤波轨迹（并不 show，仅生成 plt.Figure 返回）。
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("需要 matplotlib/numpy 库，未检测到!")
        return None
    if not os.path.isfile(csv_file):
        print(f"数据文件 {csv_file} 不存在。")
        return None
    xs_raw, ys_raw, xs_filt, ys_filt = [], [], [], []
    with open(csv_file, "r") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                x_raw = float(row["x_raw"])
                y_raw = float(row["y_raw"])
                xs_raw.append(x_raw)
                ys_raw.append(y_raw)
                if row.get("x_filt") and row.get("y_filt"):
                    xs_filt.append(float(row["x_filt"]))
                    ys_filt.append(float(row["y_filt"]))
            except Exception:
                continue

    plt.figure(figsize=(7,7))
    plt.grid(True)
    plt.title("UWB实时轨迹 (原始 vs 滤波)")

    # 原始观测点
    plt.scatter(xs_raw, ys_raw, color='orange', marker='o', s=38, alpha=0.33, label='原始定位')
    # 滤波轨迹
    if xs_filt and ys_filt:
        plt.plot(xs_filt, ys_filt, '-', color='dodgerblue', alpha=0.8, label='滤波轨迹')
        plt.scatter(xs_filt, ys_filt, facecolors='none', edgecolors='dodgerblue', marker='o', s=55, lw=1.2, alpha=0.85, label='滤波点')
    # 基站和原点
    xs_anchors = [a[0] for a in ANCHORS]
    ys_anchors = [a[1] for a in ANCHORS]
    plt.scatter([ANCHORS[0][0]], [ANCHORS[0][1]], c='black', marker='^', s=70, label='Anchor0')
    plt.scatter(xs_anchors[1:], ys_anchors[1:], c='gray', marker='^', s=70, label='Anchor1/2')
    plt.xlabel('X (米)')
    plt.ylabel('Y (米)')
    plt.legend(loc='best')
    plt.axis('equal')
    plt.tight_layout()
    # 不show，返回figure对象用于后续
    return plt.gcf()

# ========== 主程序入口 ==========
if __name__ == "__main__":
    main()