import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import scipy.signal as signal

def remove_right_step(data: np.array, show_plot=False):
    """
    除去数据画为图象图像后右边突然阶跃的部分
    """
    if show_plot:
        plt.plot(np.arange(len(data)), data)
        plt.show()

    # 二次函数变换映射
    x0 = 0.05
    A = 1/x0/x0
    diff = np.diff(data)
    test_value = np.maximum(A*(x0**2-diff**2), np.zeros(diff.shape))
    test_value = signal.medfilt(test_value, 31)  # 中值滤波
    test_value = np.where(test_value >= 0.8, 1, 0)  # 二值化
    test_value = np.diff(test_value)  # 取差分

    # 找到分割点1(一直是一段没有变化的面积)
    split_point_1_list = np.where(test_value == 1)
    if len(split_point_1_list[0]) == 0:
        print("no split_point_1")
        split_point_1 = len(data)
    else:
        split_point_1 = split_point_1_list[0][-1]
        print("split point 1:", split_point_1)

    diff = np.diff(data)  # 差分
    if show_plot:
        plt.plot(np.arange(len(diff)), diff)
        plt.show()
    split_point_2_list = np.where((np.where(diff <= -1.5, 1, 0)) == 1)  # 二值化

    # 找到分割点2(突然下降)
    if len(split_point_2_list[0]) == 0:
        print("no split_point_2")
        split_point_2 = len(data)
    else:
        split_point_2 = split_point_2_list[0][-1]
        print("split_point_2: ", split_point_2)

    # 要求分割点1&2应距离较近
    if abs(split_point_1 - split_point_2) < 20:
        print("altered data")
        split_point = min(split_point_1, split_point_2)-10
        data = data[:split_point]   # 切去有问题的部分
        if show_plot:
            plt.plot(np.arange(len(data)), data)
            plt.show()

    # 找到分割点3(突然上升)
    split_point_3_list = np.where((np.where(diff >= 5, 1, 0)) == 1)
    if len(split_point_3_list[0]) == 0:
        split_point_3 = len(data)
        print("no split_point_3")
    else:
        split_point_3 = split_point_3_list[0][-1] - 10
        print("split_point_3: ", split_point_3)
    if split_point_3 > 350:
        data = data[:split_point_3]
        if show_plot:
            plt.plot(np.arange(len(data)), data)
            plt.show()

    return data

def get_stenosis_part(data, show_plot=False):
    diff = np.diff(data)
    diff_0 = np.where(abs(diff) <= 0.001)
    minimum_area_pos = np.argmin(data)
    minimum_area = np.min(data)
    stenosis_left_pos = 0
    stenosis_right_pos = len(data)
    peaks, _ = signal.find_peaks(data)

    peaks = list(filter(lambda x: data[x]-minimum_area >= 1, peaks))
    peaks_left = list(filter(lambda x: x < minimum_area_pos, peaks))
    peaks_right = list(filter(lambda x: x > minimum_area_pos, peaks))

    for peak_pos in peaks_left[::-1]:
        var_left = np.var(data[max(0, peak_pos - 5):peak_pos])
        print(f"var_left: {var_left}")
        if var_left <= 0.006:
            stenosis_left_pos = peak_pos
            print("stenosis_left_pos: ", stenosis_left_pos)
            break

    for peak_pos in peaks_right:
        var_right = np.var(data[peak_pos+1: min(len(data), peak_pos + 6)])
        print(f"var_right: {var_right}")
        if var_right <= 0.006:
            stenosis_right_pos = peak_pos
            print("stenosis_right_pos: ", stenosis_right_pos)
            break

    if show_plot:
        plt.plot(data)
        plt.plot(diff)
        plt.plot(np.zeros(len(diff)))
        plt.plot(diff_0, np.zeros(len(diff_0)), 'r.')
        plt.plot(peaks, data[peaks], "o")
        plt.vlines([stenosis_left_pos, stenosis_right_pos], 0, np.max(data), linestyles='dashed')
        plt.vlines(minimum_area_pos, 0, np.max(data), linestyles='solid')
        plt.show()
    return data[stenosis_left_pos:stenosis_right_pos]

def get_processed_data():
    data_dir = Path("./data_out")
    for person_dir in data_dir.iterdir():
        person_id = person_dir.stem
        for csv_path in person_dir.iterdir():
            inspection_id = int(csv_path.stem.split('_')[-1]) - 1
            print("\n", person_id)
            print(csv_path.stem)
            df = pd.read_csv(csv_path)
            area = df['Area'].to_numpy()
            area = signal.medfilt(area, 7)
            area = remove_right_step(area)
            area = get_stenosis_part(area, show_plot=True)

            (data_out_dir := Path(f"./processed_data_2/{person_id:03}")).mkdir(parents=True, exist_ok=True)
            data_out_path = data_out_dir / f"ofr_area_{inspection_id}.csv"
            res_df = pd.DataFrame({"Area": area})
            res_df.to_csv(data_out_path)

get_processed_data()

