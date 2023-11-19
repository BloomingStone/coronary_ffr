import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import scipy.signal as signal


rho = 1050

def get_mid_mean(data: np.array):
    max_value = np.max(data)
    min_value = np.min(data)
    return (np.sum(data)-max_value-min_value)/(len(data)-2)

def get_S(data):
    peaks, _ = signal.find_peaks(data)
    np.insert(peaks, 0, 0)
    np.append(peaks, len(data)-1)
    stenosis, _ = signal.find_peaks(-data)
    S_list = []
    for i in range(len(peaks)-1):
        peak_left_pos, peak_right_pos = peaks[i], peaks[i+1]
        test_stenosis_part = data[peak_left_pos: peak_right_pos]
        normal_part = np.append(np.arange(peak_left_pos, peak_left_pos+3), np.arange(peak_right_pos-3, peak_right_pos))
        normal_area = get_mid_mean(data[normal_part])
        min_area = np.min(test_stenosis_part)
        S_list.append(rho / 2 * (normal_area / min_area - 1) ** 2)

    return np.mean(S_list)

def get_processed_data():
    data_dir = Path("./processed_data_2")
    for person_dir in data_dir.iterdir():
        person_id = person_dir.stem
        for csv_path in person_dir.iterdir():
            inspection_id = int(csv_path.stem.split('_')[-1]) - 1
            print("\n", person_id)
            print(csv_path.stem)
            df = pd.read_csv(csv_path)
            area = df['Area'].to_numpy()
            get_S(area, show_plot=True)

get_processed_data()

