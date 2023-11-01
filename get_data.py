from pathlib import Path
import cv2 as cv
import pandas as pd
from OfrImage import OfrImage
from threading import Thread


class MyThread(Thread):
    def __int__(self):
        Thread.__init__(self)

    def run(self):
        areas = []
        eccentricities = []
        ofr_ids = []
        for img_path in inspection_dir.iterdir():
            if img_path.suffix != ".tif":
                continue
            img = OfrImage(img_path, int(img_path.stem))
            print(f"{person_id}|{inspection_id-1}|{img.ofr_id}: {img_path}")
            output_path = ofr_out_dir / f"ofr_{img.ofr_id}.tif"
            cv.imwrite(str(output_path), img.img_recover)
            areas.append(img.area)
            eccentricities.append(img.eccentricity)
            ofr_ids.append(img.ofr_id)
        data = {'OFR_ID': ofr_ids, 'Area': areas, 'Eccentricity': eccentricities}
        df = pd.DataFrame(data)
        df.set_index(['OFR_ID'], inplace=True)
        df.sort_index(ascending=True, inplace=True)
        print(df)
        df.to_csv(data_out_path)


ofr_root = Path(r"D:\data\OFR\OFR")
"""
- ofr_root
    - person_1
        - inspection_1
            - 1.tif
            - 2.tif
"""
person_id = 0
for person_dir in ofr_root.iterdir():
    person_id = person_id + 1
    assert person_dir.is_dir()
    inspection_id = 1
    for inspection_dir in person_dir.iterdir():
        assert inspection_dir.is_dir()
        (ofr_out_dir := Path(f"./img_out/{person_id:03}/{inspection_id}")).mkdir(parents=True, exist_ok=True)
        (data_out_dir := Path(f"./data_out/{person_id:03}")).mkdir(parents=True, exist_ok=True)
        data_out_path = data_out_dir / f"ofr_area_{inspection_id}.csv"
        inspection_id = inspection_id + 1
        t = MyThread()
        t.run()
