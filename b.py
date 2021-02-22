import cv2
import time
import math
import numpy as np
from lanelines import LaneFinder
from image_calibrator import CameraCalibrator
from image_calibrator import PerspectiveTransform
from image_calibrator import RoadMarkingDetector
from maze import Maze
from maze import Cv2Aruco
from maze import Intersection


IMAGE_SHAPE = (320, 240)


def find_lane(image):
    camera_calibrator = CameraCalibrator(IMAGE_SHAPE)
    p_transformer = PerspectiveTransform((420, 240))
    marking_detector = RoadMarkingDetector()
    lf = LaneFinder((420, 240))

    image_copy = image.copy()
    # 校正鏡頭失真
    image_copy = camera_calibrator.undistort_image(image_copy)

    # 擴增影像兩邊
    image1 = cv2.copyMakeBorder(image_copy, 0, 0, 50, 50, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    image1_ = image1.copy()
    polylines_points = np.array([[[110,100], [310,100], [420,200], [0,200]]], dtype=np.int32)
    image1_ = cv2.polylines(image1_, polylines_points, True, (255, 255, 0), 2)
    cv2.imshow('image1_', image1_)

    # 透視轉換
    image2 = p_transformer.transform(image1)
    cv2.imshow('p_transformation', image2)

    # 挑出道路line (HSV)
    image3 = marking_detector.find_hsv_mask(image2, hsv_range="white")
    # cv2.imshow('image3', image3)
    # 使用滑動視窗搜索算法，查找和計算車道線像素曲線方程
    stime = time.time()
    for _ in range(10):
        sliding_window_search, slide_left_fit, slide_right_fit = lf.slide_window_search(image3, visualize=True)
    print(time.time() - stime)

    ploty = np.linspace(0, lf.height-1, lf.height)
    slide_left_fitx = np.array([])
    if slide_left_fit.size:
        slide_left_fitx = slide_left_fit[0]*ploty**2 + slide_left_fit[1]*ploty + slide_left_fit[2]
    else:
        if slide_right_fit.size:
            slide_left_fitx = (slide_right_fit[0]*ploty**2 + slide_right_fit[1]*ploty + slide_right_fit[2]) - 290
            
    slide_right_fitx = np.array([])        
    if slide_right_fit.size:
        slide_right_fitx = slide_right_fit[0]*ploty**2 + slide_right_fit[1]*ploty + slide_right_fit[2]
    else:
        if slide_left_fit.size:
            slide_right_fitx = (slide_left_fit[0]*ploty**2 + slide_left_fit[1]*ploty + slide_left_fit[2]) + 290
            # slide_right_fitx _= 420 - slide_left_fitx
    # 繪製平均方程式線條
    sliding_window_search = lf.drawing_equation(sliding_window_search, slide_left_fitx, slide_right_fitx, ploty)
    cv2.imshow('sliding_window_search', sliding_window_search)

    mean_fitx = np.mean([slide_left_fitx, slide_right_fitx], axis=0)
    if mean_fitx.size != 0:
        # angle = math.degrees(math.atan2(ploty.size - 1,
        #             mean_fitx[0] - mean_fitx[mean_fitx.size-1]))
        angle = math.degrees(math.atan2(ploty.size - 1, mean_fitx[0] - 210))
        
        # 計算曲率半徑
        curve_radius = lf.radius_of_curvature(image3, slide_left_fitx, slide_right_fitx)

        # 計算車子偏移位置
        center_offset_pix = lf.calculate_position_from_centre(image3, slide_left_fitx, slide_right_fitx)
        
    else:
        angle = 90
        curve_radius = 0
        center_offset_pix = 0

    print(f"曲線大致角度: {angle:.0f}")
    # print(f"曲率半徑: {curve_radius:.3f}(pixel)")
    towards = 'right' if center_offset_pix >= 0 else 'left'
    print(f"Vehicle is {center_offset_pix:.3f}pixel {towards} of center.")

    # 逆透視轉換
    reverse_p_transformation = p_transformer.revert(sliding_window_search)
    cv2.imshow('reverse_p_transformation', reverse_p_transformation)

    return reverse_p_transformation, angle, curve_radius, center_offset_pix


# =================== Main ==========================
if __name__ == "__main__":
    intersection = Intersection()
    my_maze = Maze()
    my_aruco = Cv2Aruco()
    
    img = cv2.imread('img/capture (9).jpg')

    image_lane, angle, _, center_offset_pix = find_lane(img)

    while True:
        

        # ===== ESC退出 =====
        if cv2.waitKey(1) & 0xff == 27:
            break

    del streaming
    cv2.destroyAllWindows()
