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
    """
    除去数据画为图象图像后右边突然阶跃的部分
    """
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
        split_point_1 = len(data)
    else:
        split_point_1 = split_point_1_list[0][-1]

    diff = np.diff(data)  # 差分
    split_point_2_list = np.where((np.where(diff <= -1.5, 1, 0)) == 1)  # 二值化

    # 找到分割点2(突然下降)
    if len(split_point_2_list[0]) == 0:
        split_point_2 = len(data)
    else:
        split_point_2 = split_point_2_list[0][-1]

    # 要求分割点1&2应距离较近
    if abs(split_point_1 - split_point_2) < 20:
        split_point = min(split_point_1, split_point_2)-10
        data = data[:split_point]   # 切去有问题的部分

    # 找到分割点3(突然上升)
    split_point_3_list = np.where((np.where(diff >= 5, 1, 0)) == 1)
    if len(split_point_3_list[0]) == 0:
        split_point_3 = len(data)
    else:
        split_point_3 = split_point_3_list[0][-1] - 10
    if split_point_3 > 350:
        data = data[:split_point_3]

    return data

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
    areas_mm2 = signal.medfilt(areas_mm2, 7)
    areas_mm2 = remove_right_step(areas_mm2)
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
