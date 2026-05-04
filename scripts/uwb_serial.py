import serial
import time
from solver import tdoa_chan
import numpy as np

ANCHORS = np.array([
        [0., 0.],
        [10., 0.],
        [0., 10.]
    ])
ANCHOR_IDS = ['A1', 'A2', 'A3']
ANCHOR_ID_TO_IDX = {aid: idx for idx, aid in enumerate(ANCHOR_IDS)}
SPEED_OF_LIGHT = 299_792_458  # m/s

def main():
    # 配置串口参数
    port = input("请输入串口号 (如 COM3 或 /dev/ttyUSB0): ").strip()
    baudrate = 115200
    timeout = 1  # 1秒超时

    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        print(f"串口 {port} 打开成功，等待数据...")
    except serial.SerialException as e:
        print(f"无法打开串口: {e}")
        return

    try:
        while True:
            data_dict = {}
            received_ids = set()
            # 连续收集一组数据（所有基站各一条）
            while len(received_ids) < len(ANCHORS):
                try:
                    line = ser.readline()
                    if not line:
                        # 超时，无数据
                        continue
                    try:
                        line_str = line.decode('utf-8').strip()
                    except UnicodeDecodeError:
                        print("解码错误，跳过此条数据")
                        continue

                    # 过滤空行
                    if not line_str:
                        continue
                    # 解析 anchor_id:timestamp_value
                    if ':' not in line_str:
                        print(f"格式错误: '{line_str}', 跳过")
                        continue
                    anchor_id, ts_val = line_str.split(':', 1)
                    anchor_id = anchor_id.strip()
                    ts_val = ts_val.strip()
                    if anchor_id not in ANCHOR_ID_TO_IDX:
                        print(f"未知基站ID: '{anchor_id}', 跳过")
                        continue
                    idx = ANCHOR_ID_TO_IDX[anchor_id]
                    try:
                        ts_val = float(ts_val)
                    except ValueError:
                        print(f"时间戳值无法转换为float: '{ts_val}', 跳过")
                        continue
                    if anchor_id in data_dict:
                        # 已有此基站数据，跳过或覆盖
                        print(f"重复基站 {anchor_id} 数据，覆盖前值")
                    data_dict[anchor_id] = ts_val
                    received_ids.add(anchor_id)
                except serial.SerialException as e:
                    print(f"串口错误: {e}")
                    time.sleep(1)
                except Exception as e:
                    print(f"未知错误: {e}")

            # 按anchor_id顺序构建timestamps数组
            try:
                time_stamps = np.array([data_dict[aid] for aid in ANCHOR_IDS])
            except KeyError as e:
                print(f"缺失基站数据: {e}, 放弃此组定位")
                continue

            # 以第一个基站为参考
            ref_time = time_stamps[0]
            time_diffs = time_stamps[1:] - ref_time

            try:
                coord = tdoa_chan(ANCHORS, time_diffs, speed_of_light=SPEED_OF_LIGHT)
                print(f"定位结果: x={coord[0]:.4f}, y={coord[1]:.4f}")
            except Exception as e:
                print(f"定位计算异常: {e}")
            # 短暂等待，避免刷屏过快
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n用户中断，关闭串口...")
    finally:
        ser.close()
        print("串口已关闭")

if __name__ == "__main__":
    main()
    