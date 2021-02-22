import cv2
import time
import math
import numpy as np
import sys
import signal
import socket
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw
from lanelines import LaneFinder
from image_calibrator import CameraCalibrator
from image_calibrator import PerspectiveTransform
from image_calibrator import RoadMarkingDetector


def plot_images_2x3(images):
    images_len = len(images)
    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    for i in range(0, images_len):
        ax = plt.subplot(2, 3, 1+i)
        ax.imshow(images[i])
        ax.axis('off')
    plt.subplots_adjust(left=0., right=1, top=0.7, bottom=0.)
    plt.show()


# camera_calibrator = CameraCalibrator((320, 240))
# img1 = cv2.imread("./img/route/route (2).jpg")
# img1 = camera_calibrator.undistort_image(img1)
# img2 = cv2.imread("./img/route/route (3).jpg")
# img2 = camera_calibrator.undistort_image(img2)
# img3 = cv2.imread("./img/route/route (4).jpg")
# img3 = camera_calibrator.undistort_image(img3)
# img4 = cv2.imread("./img/route/route (5).jpg")
# img4 = camera_calibrator.undistort_image(img4)
# plot_images_2x3([img1, img2, img3, img4])



image = cv2.imread("./img/aruco/aruco (10).jpg")


camera_calibrator = CameraCalibrator((320, 240))
p_transformer = PerspectiveTransform((420, 240))
marking_detector = RoadMarkingDetector()
lf = LaneFinder((420, 240))

image_copy = image.copy()
# 校正鏡頭失真
image_copy = camera_calibrator.undistort_image(image_copy)

# 擴增影像兩邊
image1 = cv2.copyMakeBorder(image_copy, 0, 0, 50, 50, cv2.BORDER_CONSTANT, value=(0,0,0))

image1_ = image1.copy()
polylines_points = np.array([[[110,100],[310,100],[420,200],[0,200]]], dtype=np.int32)
image1_ = cv2.polylines(image1_, polylines_points, True, (255, 255, 0), 2)
cv2.imshow('image1_', image1_)

# 透視轉換
image2 = p_transformer.transform(image1)
# cv2.imshow('p_transformation', image2)

# 挑出道路line (HSV)
image3 = marking_detector.find_hsv_mask(image2, hsv_range="white")
cv2.imshow('image3', image3)
# 使用滑動視窗搜索算法，查找和計算車道線像素曲線方程
sliding_window_search, slide_left_fit, slide_right_fit = lf.slide_window_search(image3, visualize=True)

ploty = np.linspace(0, lf.height-1, lf.height)
slide_left_fitx = np.array([])
if slide_left_fit.size:
    slide_left_fitx = slide_left_fit[0]*ploty**2 + slide_left_fit[1]*ploty + slide_left_fit[2]
else:
    if slide_right_fit.size:
        slide_left_fitx = (slide_right_fit[0]*ploty**2 + slide_right_fit[1]*ploty + slide_right_fit[2]) - 340
        
slide_right_fitx = np.array([])        
if slide_right_fit.size:
    slide_right_fitx = slide_right_fit[0]*ploty**2 + slide_right_fit[1]*ploty + slide_right_fit[2]
else:
    if slide_left_fit.size:
        slide_right_fitx = (slide_left_fit[0]*ploty**2 + slide_left_fit[1]*ploty + slide_left_fit[2]) + 340
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
# print(f"Vehicle is {abs(center_offset_pix):.3f}pixel {towards} of center.")

# 逆透視轉換
reverse_p_transformation = p_transformer.revert(sliding_window_search)

cv2.imshow('reverse_p_transformation', reverse_p_transformation)




# cv2.imshow('image_lane', image_lane)

cv2.waitKey(0)




# camera_calibrator = CameraCalibrator((320, 240))
# p_transformer = PerspectiveTransform((420, 240), src_vector=[[150,70],[270,70],[420,240],[0,240]])

# img = cv2.imread("./img/route/route (4).jpg")

# # 校正鏡頭失真
# img = camera_calibrator.undistort_image(img)

# # 擴增影像兩邊
# img = cv2.copyMakeBorder(img, 0, 0, 50, 50, cv2.BORDER_CONSTANT, value=(0,0,0))
# img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # 透視轉換
# # image_trans = p_transformer.transform(img)
# image_trans = img
# # cv2.imshow('image_trans', image_trans)
# image_trans_ = cv2.cvtColor(image_trans, cv2.COLOR_BGR2RGB)

# height, width = image_trans_.shape[:2]
# row = height // 3
# col = width // 3
# cv2.line(image_trans_, (col, 0), (col, height), (255,255,0), 2)
# cv2.line(image_trans_, (col*2, 0), (col*2, height), (255,255,0), 2)
# cv2.line(image_trans_, (0, row), (width, row), (255,255,0), 2)
# cv2.line(image_trans_, (0, row*2), (width, row*2), (255,255,0), 2)


# # 對影像中的紅色做遮罩
# hsv = cv2.cvtColor(image_trans, cv2.COLOR_BGR2HSV)
# lower_red = np.array([170,10,200])
# upper_red = np.array([180,255,255])
# mask1 = cv2.inRange(hsv, lower_red, upper_red)
# lower_red = np.array([0,10,200])
# upper_red = np.array([10,255,255])
# mask2 = cv2.inRange(hsv, lower_red, upper_red)
# thresh = cv2.bitwise_or(mask1,mask2)
# # cv2.imshow('thresh', thresh)


# (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(image_trans_, cnts, -1, (0, 255, 0), 2)

# test_img = image_trans_.copy()

# for c in cnts:        
#     mask = np.zeros(thresh.shape, dtype="uint8")
#     cv2.drawContours(mask, [c], -1, 255, -1)
#     test_img = cv2.bitwise_and(image_trans_, image_trans_, mask=mask)
    
#     M = cv2.moments(c)
#     cX = int(M["m10"] / M["m00"])
#     cY = int(M["m01"] / M["m00"])
#     cv2.circle(test_img, (cX, cY), 10, (1, 227, 254), -1)
    
#     cv2.imshow("Image + Mask", test_img)
#     cv2.waitKey(0)


# output = cv2.bitwise_and(image_trans, image_trans, mask=thresh)
# front_val = np.sum(thresh[:row, :width] == 255)
# left_val = np.sum(thresh[:height, :col] == 255)
# right_val = np.sum(thresh[:height, col*2:width] == 255)

# print(f"前方: {front_val}")
# print(f"左方: {left_val}")
# print(f"右方: {right_val}")
# cv2.putText(output, str(front_val), (180,70), 0, 1, (0,255,0), 2)
# cv2.putText(output, str(left_val), (30,130), 0, 1, (0,255,0), 2)
# cv2.putText(output, str(right_val), (290,130), 0, 1, (0,255,0), 2)


# plot_images_2x3([img_, image_trans_, thresh, output])

# cv2.waitKey(0)



# def plot_images1(images, idx=0, num=10):
#     fig = plt.gcf()
#     fig.set_size_inches(10, 8)
#     if num > 16:
#         num = 16
#     for i in range(0, num):
#         ax = plt.subplot(4, 4, 1+i)
#         ax.imshow(images[idx+i])
#         ax.axis('off')
#     plt.show()


# def plus(*nums):
#     res = 0
#     for i in nums:
#         res += i
#     return res

# a = plus(1,2,3)


# def plot_images_2x3(images):
#     images_len = len(images)
#     fig = plt.gcf()
#     fig.set_size_inches(12, 6)
#     for i in range(0, images_len):
#         ax = plt.subplot(2, 3, 1+i)
#         image = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
#         ax.imshow(image)
#         ax.axis('off')
#     plt.show()

# img1 = cv2.imread("./img/capture (1).jpg")
# img2 = cv2.imread("./img/capture (4).jpg")
# img3 = cv2.imread("./img/capture (7).jpg")
# img4 = cv2.imread("./img/capture (9).jpg")
# img5 = cv2.imread("./img/capture (11).jpg")
# plot_images_2x3([img1, img2, img3, img4, img5])

