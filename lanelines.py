import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# class Line():
#     def __init__(self, image_shape):
#         self.image_shape = image_shape  # w, h


class LaneFinder(object):
    """ 在影像中找出車道 """
    def __init__(self, image_shape):
        self.image_shape = image_shape  # w, h
        self.width = image_shape[0]
        self.height = image_shape[1]
    
    def slide_window_search(self, binary_image, visualize=False):
        """使用滑動視窗搜索算法來查找和計算車道線像素曲線方程

        Args:
            binary_image: (numpy.ndarray)影像
            visualize: (bool)是否繪製線條、視窗
        Returns:
            out_img: (numpy.ndarray)輸出影像
            left_fitx: (numpy.ndarray)左線條方程式
            right_fitx: (numpy.ndarray)右線條方程式
            ploty: (numpy.ndarray)Y軸
        """
        # binary_warped = warped_image.astype(np.uint8)
        width, height = self.width, self.height
        # 滑動視窗的數量
        nwindows = 9
        # 滑動視窗的高度
        window_height = np.int(height//nwindows)
        # 圖像像素的直方圖
        histogram = np.sum(binary_image[int(height-60):,:], axis=0) // 255
        # plt.figure()
        # plt.plot(histogram)
        # plt.show()

        midpoint = np.int(width//2)
        leftx_base = np.argmax(histogram[:midpoint])
        leftx_max = np.max(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        rightx_max = np.max(histogram[midpoint:])
        
        # 識別圖像中所有非零像素的x和y位置
        nonzero = binary_image.nonzero()
        # 所有非零像素的x和y坐標
        nonzeroy = np.array(nonzero[0])   
        nonzerox = np.array(nonzero[1])

        # 每個視窗的當前位置要更新
        leftx_current = leftx_base
        rightx_current = rightx_base

        windows = [] # 記錄可視化的搜索窗口
        margin = 80 # 設置視窗的寬度+/-邊距
        margin_l = 80 
        margin_r = 80 
        minpix = 30 # 設置找到重定位視窗的最小像素數

        have_left_line = True if (leftx_max > 20) else False
        have_right_line = True if (rightx_max > 20) else False

        # 創建空列表以接收左右車道像素索引
        left_lane_inds = []
        right_lane_inds = []
        
        for window in range(nwindows):
            win_y_low = height - (window+1)*window_height
            win_y_high = height - window*window_height
            win_xleft_low = leftx_current - margin_l
            win_xleft_high = leftx_current + margin_l
            win_xright_low = rightx_current - margin_r
            win_xright_high = rightx_current + margin_r
            
            # 紀錄搜索視窗
            if have_left_line:
                windows.append((win_xleft_low, win_y_low, win_xleft_high, win_y_high))
            if have_right_line:
                windows.append((win_xright_low, win_y_low, win_xright_high, win_y_high))

            # 識別視窗中x和y的非零像素
            if have_left_line:
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            else:
                good_left_inds = []
            if have_right_line:
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            else:
                good_right_inds = []
            
            # 將這些索引附加到列表中
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # 如果找到>minpix像素，則在其平均位置重新顯示下一個視窗
            if len(good_left_inds) > minpix:
                last_leftx_current = leftx_current
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                margin_l = abs(leftx_current - last_leftx_current) * 2 + 30
            if len(good_right_inds) > minpix:
                last_rightx_current = rightx_current
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                margin_r = abs(rightx_current - last_rightx_current) * 2 + 30
        
        # 連接索引數組
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        # 提取左右行像素位置
        if have_left_line:
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
        else:
            leftx = []
            lefty = []

        if have_right_line:
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
        else:
            rightx = []
            righty = []

        # 為每個擬合二階多項式
        if have_left_line:
            left_fit = np.polyfit(lefty, leftx, 2)
        else:
            left_fit = np.array([])

        if have_right_line:
            right_fit = np.polyfit(righty, rightx, 2)
        else:
            right_fit = np.array([])

        # 創建一個輸出圖像（3個通道）以進行繪製並可視化結果
        out_img = None
        if visualize:
            out_img = np.dstack((binary_image, binary_image, binary_image))*255
            # 左右線條上色
            if have_left_line:
                out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            if have_right_line:
                out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            # 繪製視窗
            for rect in windows:
                tlx, tly, brx, bry = rect
                cv2.rectangle(out_img, (tlx, tly), (brx, bry), (0, 255, 0), 2)
            '''
            ploty = np.linspace(0, height-1, height)
            if left_fit.size:
                left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
                self.drawing_equation(out_img, left_fitx, left_fitx, ploty)
                _left_fitx = left_fitx + 100
                self.drawing_equation(out_img, _left_fitx, _left_fitx, ploty)
                   
            if right_fit.size:
                right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
                self.drawing_equation(out_img, right_fitx, right_fitx, ploty)
                _right_fitx = right_fitx - 100
                self.drawing_equation(out_img, _right_fitx, _right_fitx, ploty)
            '''
        else:
            out_img = binary_image

        return out_img, left_fit, right_fit
        
    def calculate_radious_of_curvature(self, length, fitx):
        """計算曲率半徑

        Args:
            length: (int)長度(影像高)
            fitx: (numpy.ndarray)線條方程式
        Returns:
            curve_radius: (numpy.float64)曲率半徑
        """
        y_m_per_pix = 1 #30/720    # Meters per pixel in y dimension
        x_m_per_pix = 1 #3.7/700   # Meters per pixel in x dimension
        
        plot_y = np.linspace(0, length-1, length)   # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(plot_y*y_m_per_pix, fitx*x_m_per_pix, 2)
        y_eval = np.max(plot_y) # Calculate the new radii of curvature
        curve_radius = ((1 + (2*fit_cr[0]*y_eval*y_m_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        return curve_radius

    def radius_of_curvature(self, image, left_fitx, right_fitx):
        """計算平均曲率半徑

        Args:
            image: (numpy.ndarray)影像
            left_fitx: (numpy.ndarray)左線條方程式
            right_fitx: (numpy.ndarray)右線條方程式
        Returns:
            curve_radius: (numpy.float64)平均曲率半徑
        """
        left_curve_radius = self.calculate_radious_of_curvature(image.shape[0], left_fitx)
        right_curve_radius = self.calculate_radious_of_curvature(image.shape[0], right_fitx)
        curve_radius = np.mean([left_curve_radius, right_curve_radius])
        # cv2.putText(image,f"Radius of Curvature = {curve_radius:.3f}(m)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        return curve_radius

    def calculate_position_from_centre(self, image, left_fitx, right_fitx):
        """計算車子偏移位置

        Args:
            image: (numpy.ndarray)影像
            left_fitx: (numpy.ndarray)左線條方程式
            right_fitx: (numpy.ndarray)右線條方程式
        Returns:
            center_offset_mtrs: (int)車子偏移距離
        """
        lane_center = (right_fitx[image.shape[0]-1] + left_fitx[image.shape[0]-1])/2
        x_m_per_pix = 1 #3.7/700   # Meters per pixel in x dimension
        
        center_offset_pixels = image.shape[1]/2 - lane_center
        center_offset_mtrs = x_m_per_pix*center_offset_pixels
        # cv2.putText(image,'Vehicle is {0:.3f}m '.format(abs(center_offset_mtrs)) + towards +' of center.',(50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        return center_offset_mtrs

    def drawing_equation(self, image, left_fitx, right_fitx, ploty):
        """繪製平均方程式線條

        Args:
            image: (numpy.ndarray)影像
            left_fitx: (numpy.ndarray)左線條方程式
            right_fitx: (numpy.ndarray)右線條方程式
            ploty: (numpy.ndarray)
        Returns:
            image: (numpy.ndarray)輸出影像
        """
        mean_fitx = np.mean([left_fitx, right_fitx], axis=0)
        mean_fitx_point = [(int(x), int(y)) for y, x in zip(ploty, mean_fitx)]
        for point in mean_fitx_point:
            cv2.circle(image, point, 1, (255,255,0), 1)
        return image
