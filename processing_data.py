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

    right_end_data = data[-1]
    diff_to_right_end_square = (data-right_end_data)**2
    diff_after_filter = signal.medfilt(diff_to_right_end_square, 5)
    split_point = np.where(np.diff(np.where(diff_after_filter < 0.05, 1, 0))==1)[0][-1]
    if len(data)-split_point > 2:
        data = data[:(split_point-5)]

    # if show_plot:
    #     plt.plot(np.arange(len(data)), data)
    #     plt.show()

    data_after_filter = signal.medfilt(data, 7)
    diff = np.diff(data_after_filter)
    diff_sum = diff[1:]+diff[:-1]

    plt.plot(diff_sum)
    plt.show()

    # 找到分割点(突然上升)
    split_point_2_list = np.where((np.where(diff_sum >= 3, 1, 0)) == 1)[0]
    if len(split_point_2_list) != 0:
        split_point_2 = split_point_2_list[-1]-5
        data = data[:split_point_2]
        if show_plot:
            plt.plot(np.arange(len(data)), data)
            plt.show()

    return data

def replace_outliers_with_mean(series):
    result = series.copy()

    for i in range(len(series)):
        # 计算周围11个数（包括自己）的均值和方差
        start_index = max(0, i - 5)
        end_index = min(len(series), i + 6)
        neighborhood = series[start_index:end_index]

        mean_value = np.mean(neighborhood)
        sqrt_std_value = np.sqrt(np.std(neighborhood))

        # 判断是否离均值相距大于两个标准差
        if np.abs(series[i] - mean_value) > 2 * sqrt_std_value:
            # 用均值替代
            result[i] = mean_value

    return result


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
            area = remove_right_step(area, show_plot=True)
            # area = get_stenosis_part(area, show_plot=True)

            # plt.plot(area)
            # plt.show()
            # area = replace_outliers_with_mean(area)
            # plt.plot(area)
            # plt.show()

            # (data_out_dir := Path(f"./processed_data_2/{person_id:03}")).mkdir(parents=True, exist_ok=True)
            # data_out_path = data_out_dir / f"ofr_area_{inspection_id}.csv"
            # res_df = pd.DataFrame({"Area": area})
            # res_df.to_csv(data_out_path)

get_processed_data()

