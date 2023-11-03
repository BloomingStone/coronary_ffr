import pandas as pd
import math
import numpy as np
from pathlib import Path

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

    def get_f_s(area: np.array):
        An = max(area[0, 0], area[-1, 0])
        As = area.min()

        F_ = 8 * math.pi * mu * L_OCT * np.sum(An / area / area)
        S_ = rho / 2 * (An / As - 1) ** 2
        F_mmHg_s_div_cm = F_ * 0.0075 * 0.01
        S_mmHg_s2_div_cm2 = S_ * 0.0075 * 0.01 * 0.01
        return F_mmHg_s_div_cm, S_mmHg_s2_div_cm2

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

    data_dir = Path(r"./processed_data")
    person_id = 0
    FFRs = []
    for person_dir in data_dir.iterdir():
        person_id = int(person_dir.stem)-1
        V = left_v if ofr_types[person_id] == 'L' else right_v
        inner_list = []
        for csv_path in person_dir.iterdir():
            inspection_id = int(csv_path.stem.split('_')[-1])-1
            df = pd.read_csv(csv_path)

            F, S = get_f_s(df[['Area']].to_numpy()/1000000)

            dP = dict(map(get_delta_p_mapper(F, S, V), ["dia", "sys"]))
            FFR = (2/3*(P["dia"] - dP["dia"]) + 1/3*(P["sys"] - dP["sys"]))/P["mean"]
            print("{}_{}: FFR={:.3} mmHg".format(person_id+1, inspection_id+1, FFR))
            inner_list.append(FFR)
        FFRs.append(np.mean(inner_list))

    [print(ffr) for ffr in FFRs]
    return FFRs


calculate_all()