import math
import click
import numpy as np
from pathlib import Path
import cv2 as cv
from scipy import signal
from threading import Thread

mu = 4.0e-3
rho = 1050
L_OCT = 0.18e-3  # OCT切片相隔距离, 0.18mm

left_v = {"dia": 20, "sys": 10}  # diastolic systolic
right_v = {"dia": 15, "sys": 10}
P = {"dia": 60, "sys": 120, "mean": 80}
max_SFR = {"dia": 4.2, "sys": 2.0}

def get_recovered_img(img_origin):
    flags = cv.INTER_LINEAR | cv.WARP_FILL_OUTLIERS | cv.WARP_INVERSE_MAP
    res = cv.warpPolar(img_origin, (1024, 1024), (512, 512), 512, flags)
    return res

def get_biggest_contour(contours):
    max_area = 0
    res = contours[0]
    for contour in contours:
        area = cv.contourArea(contour)
        if max_area < area:
            max_area = area
            res = contour
    return res

def get_s(data):
    peaks, _ = signal.find_peaks(data)
    np.insert(peaks, 0, 0)
    np.append(peaks, len(data) - 1)
    stenosis, _ = signal.find_peaks(-data)
    S_list = []
    for i in range(len(peaks) - 1):
        peak_left_pos, peak_right_pos = peaks[i], peaks[i + 1]
        test_stenosis_part = data[peak_left_pos: peak_right_pos]
        normal_area = max(data[peak_left_pos], data[peak_right_pos])
        min_area = np.min(test_stenosis_part)
        S_list.append(rho / 2 * (normal_area / min_area - 1) ** 2)

    return np.sum(S_list)

def remove_right_step(data: np.array):
    # 去除导引导管区域
    res = data.copy()
    right_end_data = res[-1]    # 取最右端的值作为参考, 去除所有最右端面积相近处的值(导引导管)
    diff_to_right_end_square = (res-right_end_data)**2
    diff_after_filter = signal.medfilt(diff_to_right_end_square, 5)  # 导引区域面积有时候会突然增加, 过滤掉
    split_point = np.where(np.diff(np.where(diff_after_filter < 0.05, 1, 0)) == 1)[0][-1]
    if len(res)-split_point > 2:
        res = res[:(split_point-5)]  # 向左边多取一点, 保证导引区域全部去除, 下同

    # 去除主动脉区域(突然上升)
    data_after_filter = signal.medfilt(res, 7)
    diff = np.diff(data_after_filter)
    diff_sum = diff[1:]+diff[:-1]  # 每两个相邻diff相加, 是连续增加

    split_point_2_list = np.where((np.where(diff_sum >= 4, 1, 0)) == 1)[0]
    if len(split_point_2_list) != 0:
        split_point_2 = split_point_2_list[-1]-5
        res = res[:split_point_2]

    return res

def get_delta_p(F, S, V_normal):
    res = {}
    for heart_status in ['dia', 'sys']:
        b = F + 4.5 / max_SFR[heart_status]
        SFR = (math.sqrt(b ** 2 + 360 * S) - b) / 40 / S
        dP = F * V_normal[heart_status] * SFR + S * (V_normal[heart_status] * SFR) ** 2
        res.update({heart_status: dP})
    return res


def get_ffr(tif_dir, coronary_side: str):
    """
    输入tif分割结果文件所在的目录, 打印ffr
    """
    if coronary_side is str:
        tif_dir = Path(tif_dir)
        if not tif_dir.is_absolute():
            tif_dir = Path.cwd() / tif_dir
    ofr_id_area_map = {}
    assert tif_dir.is_dir()
    for img_path in tif_dir.iterdir():
        if img_path.suffix != ".tif":
            continue
        ofr_id = int(img_path.stem)
        img_origin = cv.imread(str(img_path), cv.IMREAD_UNCHANGED)
        img_recover = get_recovered_img(img_origin)
        contours = cv.findContours(img_recover, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
        biggest_contour = get_biggest_contour(contours)
        max_area_mm2 = cv.contourArea(biggest_contour)/102.4/102.4  # 10mm * 10mm 1024*1204 像素
        ofr_id_area_map.update({ofr_id: max_area_mm2})

    areas_mm2 = [ofr_id_area_map[i] for i in sorted(ofr_id_area_map)]
    areas_mm2 = np.array(areas_mm2)
    areas_mm2 = remove_right_step(areas_mm2)
    areas_mm2 = signal.medfilt(areas_mm2, 5)
    F = 8 * math.pi * mu * L_OCT * np.sum(1 / areas_mm2) * 1e6
    S = get_s(areas_mm2)
    F_mmHg_s_div_cm = F * 0.0075 * 0.01
    S_mmHg_s2_div_cm2 = S * 0.0075 * 0.01 * 0.01

    assert coronary_side == 'L' or coronary_side == 'R'
    V_normal = left_v if coronary_side == 'L' else right_v
    dP = get_delta_p(F_mmHg_s_div_cm, S_mmHg_s2_div_cm2, V_normal)
    FFR = (2 * (P["dia"] - dP["dia"]) + (P["sys"] - dP["sys"])) / 3 / P["mean"]
    return FFR


class GetFfrThread(Thread):
    def __init__(self, person_id, tif_dir, coronary_side):
        super().__init__()
        self.person_id = person_id
        self.tif_dir = tif_dir
        self.coronary_side = coronary_side
        self.res = None

    def run(self) -> None:
        self.res = get_ffr(self.tif_dir, self.coronary_side)

@click.command()
@click.option('-d', '--root_dir', default='', help="""
输入存放多人分割结果的文件夹路径, 一次打印ffr结果
""")
def get_multi_ffr(root_dir):
    """
    输入存放多人分割结果的文件夹路径, 一次打印ffr结果
    文件夹组织方式为
    - ofr_root
    - person_(L/R)
        - inspection_1
            - 1.tif
            - 2.tif
    """
    if isinstance(root_dir, str):
        root_dir = Path(root_dir)
        if not root_dir.is_absolute():
            root_dir = Path.cwd() / root_dir

    t_list = []
    for person_id, person_dir in enumerate(root_dir.iterdir()):
        assert person_dir.is_dir()
        coronary_side = str(person_dir).split('_')[-1]
        for inspection_dir in person_dir.iterdir():
            assert inspection_dir.is_dir()
            t = GetFfrThread(person_id, inspection_dir, coronary_side)
            t.start()
            t_list.append(t)

    ffr_dict = {}
    for t in t_list:
        t.join()
        person_id = t.person_id
        if person_id not in ffr_dict:
            ffr_dict.update({person_id: t.res})
        else:
            ffr_dict.update({person_id: 0.5*(ffr_dict[person_id] + t.res)})

    for i in sorted(ffr_dict):
        print(ffr_dict[i])


if __name__ == '__main__':
    get_multi_ffr()
