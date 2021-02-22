import os
import cv2
import numpy as np
from glob import glob
# from tqdm import tqdm


DISTORTION_MATRIX_FILE = "camera_matrix.npy"


class CameraCalibrator(object):
    """鏡頭校正

    Args:
        image_shape: (tuple)影像大小(w, h)
    """
    def __init__(self, image_shape):
        camera_matrix = np.load(DISTORTION_MATRIX_FILE)
        self.image_shape = image_shape  # w, h
        self.mtx = camera_matrix.item().get('mtx')
        self.dist = camera_matrix.item().get('dist')
        self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(self.mtx, 
                                                self.dist, self.image_shape, 1)

    def undistort_image(self, img):
        """校正鏡頭失真

        Args:
            img: (numpy.ndarray)失真影像
        Returns:
            undistorted: (numpy.ndarray)校正後影像
        """
        dst = cv2.undistort(img, self.mtx, self.dist, None, self.newcameramtx)
        # dst = cv2.undistort(img, mtx, dist, None, mtx)
        # crop the image
        # roi_x, roi_y, roi_w, roi_h = self.roi
        # dst2 = dst[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

        dst = dst[15:225, 15:305]
        dst2 = cv2.resize(dst, (320,240))

        undistorted = cv2.resize(dst2, self.image_shape, interpolation=cv2.INTER_CUBIC)
        return undistorted


class PerspectiveTransform(object):
    """透視轉換

    Args:
        image_shape: (tuple)影像大小(w, h)
        src_vector: (list)透視矩陣
    """
    def __init__(self, image_shape, src_vector=[[110,100],[310,100],[420,200],[0,200]]):
        self.image_shape = image_shape
        width, height = image_shape
        src = np.float32(src_vector)
        dst = np.float32([[0,0],[width,0],[width,height],[0,height]])

        self.transform_matrix = cv2.getPerspectiveTransform(src, dst)
        self.revert_matrix = cv2.getPerspectiveTransform(dst, src)

    def transform(self, img):
        """ 透視轉換 """
        return self._warp_perspective(img, self.transform_matrix)

    def revert(self, img):
        """ 透視還原 """
        return self._warp_perspective(img, self.revert_matrix)

    def _warp_perspective(self, img, matrix):
        return cv2.warpPerspective(img, matrix, self.image_shape,
                                   flags=cv2.INTER_LINEAR)


class RoadMarkingDetector(object):
    def __init__(self):
        self.lower_white = np.array([0,0,170])
        self.upper_white = np.array([179,150,255])
        self.lower_red_1 = np.array([0,30,200])
        self.upper_red_1 = np.array([10,255,255])
        self.lower_red_2 = np.array([170,30,200])
        self.upper_red_2 = np.array([179,255,255])

    def find_hsv_mask(self, image, hsv_range="white"):
        """對影像HSV依閾值二值化

        Args:
            image: (numpy.ndarray)影像
            hsv_range: (string or list[numpy.array*2])需要的 顏色 或 hsv範圍
        Returns:
            binary_img: (numpy.ndarray)輸出影像(二值圖)
        """
        if isinstance(hsv_range, str):
            if hsv_range == "white":
                lower = self.lower_white
                upper = self.upper_white
            elif hsv_range == "red1":
                lower = self.lower_red_1
                upper = self.upper_red_1
            elif hsv_range == "red2":
                lower = self.lower_red_2
                upper = self.upper_red_2
            else:
                lower = np.array([0,0,0])
                upper = np.array([179,255,255])
        else:
            lower = hsv_range[0]
            upper = hsv_range[1]

        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        binary_img = cv2.inRange(hsv_img, lower, upper)
        return binary_img
