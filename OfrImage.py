from pathlib import Path
import cv2 as cv
import numpy as np


class OfrImage:
    ID = 0

    def __init__(self, path: Path, ofr_id=None):
        if ofr_id is None:
            OfrImage.ID = OfrImage.ID + 1
            self.ofr_id = OfrImage.ID
        else:
            OfrImage.ID = ofr_id
            self.ofr_id = ofr_id
        self.path = path
        self.img_origin = cv.imread(str(self.path), cv.IMREAD_UNCHANGED)
        self.img_recover = self.get_recovered_img()
        self.contours = cv.findContours(self.img_recover, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
        self.biggest_contour = self.get_biggest_contour()
        self.area = cv.contourArea(self.biggest_contour)/102.4/102.4*1.8
        self.area_non_zero = cv.countNonZero(self.img_recover)/102.4/102.4*1.8
        self.eccentricity = self.get_eccentricity()

    def get_recovered_img(self):
        flags = cv.INTER_LINEAR | cv.WARP_FILL_OUTLIERS | cv.WARP_INVERSE_MAP
        res = cv.warpPolar(self.img_origin, (1024, 1024), (512, 512), 512, flags)
        return res

    def get_biggest_contour(self):
        max_area = 0
        res = self.contours[0]
        for contour in self.contours:
            area = cv.contourArea(contour)
            if max_area < area:
                max_area = area
                res = contour
        return res

    def get_eccentricity(self):
        """
        得到最大连通域的偏心率
        """
        ellipse = cv.fitEllipse(self.biggest_contour)
        _, (a, b), ang = np.int32(ellipse[0]), np.int32(ellipse[1]), round(ellipse[2], 1)
        if a > b:
            eccentric = np.sqrt(1.0 - (b / a) ** 2)  # a 为长轴
        else:
            eccentric = np.sqrt(1.0 - (a / b) ** 2)
        return eccentric

    def show_contours(self):
        result = np.zeros_like(self.img_recover)
        cv.drawContours(result, self.contours, -1, [255, ], 1)
        cv.imshow('Contours', result)
        cv.waitKey(0)
