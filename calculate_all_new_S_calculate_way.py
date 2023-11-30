import pandas as pd
import math
import numpy as np
from pathlib import Path

from matplotlib import pyplot as plt
from scipy import signal

def calculate_all():
    # 设置常数, 所有数据单位都是基本单位
    mu = 4.0e-3
    rho = 1050
    k = 1
    L_OCT = 0.18e-3  # OCT切片相隔距离, 0.18mm

    left_v = {"dia": 20, "sys": 10}  # diastolic systolic
    right_v = {"dia": 15, "sys": 10}
    P = {"dia": 60, "sys": 120, "mean": 80}
    max_SFR = {"dia": 4.2, "sys": 2.0}

    def get_S(data):
        peaks, _ = signal.find_peaks(data)
        np.insert(peaks, 0, 0)
        np.append(peaks, len(data) - 1)
        stenosis, _ = signal.find_peaks(-data)
        plt.plot(data)
        plt.vlines(peaks, min(data), max(data), 'r', linestyles='dashed')
        plt.plot(stenosis, data[stenosis], '*')
        plt.show()
        S_list = []
        for i in range(len(peaks) - 1):
            peak_left_pos, peak_right_pos = peaks[i], peaks[i + 1]
            test_stenosis_part = data[peak_left_pos: peak_right_pos]
            normal_area = max(data[peak_left_pos] , data[peak_right_pos])
            min_area = np.min(test_stenosis_part)
            S_list.append(rho / 2 * (normal_area / min_area - 1) ** 2)

        return np.sum(S_list)

    def get_delta_p_mapper(F_, S_, V_):
        def inner(condition):
            assert condition == "dia" or condition == "sys"
            b = F_ + 4.5 / max_SFR[condition]
            SFR = (math.sqrt(b ** 2 + 360 * S_) - b) / 40 / S_
            Delta_P = F_ * V_[condition] * SFR + S_ * (V_[condition] * SFR) ** 2
            return condition, Delta_P
        return inner

    dcm_root = Path(r"D:\data\OFR\OCT OFR DCM")
    ofr_types = [dcm.stem[4:5] for dcm in dcm_root.iterdir()]  # L or R

    data_dir = Path(r"./processed_data_2")
    person_id = 0
    FFRs = []
    for person_dir in data_dir.iterdir():
        person_id = int(person_dir.stem)-1
        V = left_v if ofr_types[person_id] == 'L' else right_v
        inner_list = []
        for csv_path in person_dir.iterdir():
            inspection_id = int(csv_path.stem.split('_')[-1])-1
            df = pd.read_csv(csv_path)

            area_mm2 = df['Area'].to_numpy()

            F = 8 * math.pi * mu * L_OCT * np.sum(1 / area_mm2)*1e6
            S = get_S(area_mm2)
            F_mmHg_s_div_cm = F * 0.0075 * 0.01
            S_mmHg_s2_div_cm2 = S * 0.0075 * 0.01 * 0.01

            dP = dict(map(get_delta_p_mapper(F_mmHg_s_div_cm, S_mmHg_s2_div_cm2, V), ["dia", "sys"]))
            FFR = (2/3*(P["dia"] - dP["dia"]) + 1/3*(P["sys"] - dP["sys"]))/P["mean"]
            print("{}_{}: FFR={:.3} mmHg".format(person_id+1, inspection_id+1, FFR))
            inner_list.append(FFR)
        FFRs.append(np.mean(inner_list))

    [print(ffr) for ffr in FFRs]
    return FFRs


calculate_all()