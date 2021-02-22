import cv2
import time
import math
import sys
import signal
import socket
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw
from lanelines import LaneFinder
from image_calibrator import CameraCalibrator
from image_calibrator import PerspectiveTransform
from image_calibrator import RoadMarkingDetector
from maze import Maze
from maze import Cv2Aruco
from maze import Intersection


TCP_IP = "192.168.0.192"
# TCP_IP = "192.168.43.5"
TCP_PORT = 8003
IMAGE_SHAPE = (320, 240)


# 中斷
def signal_handler(signal, frame):
    global interrupted
    interrupted = True
    conn.close()
    s.close()
    cv2.destroyAllWindows()
    print('Stop pressing the CTRL+C!')
    sys.exit(0)

def interrupt_callback():
    global interrupted
    return interrupted


def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def plot_images_2x3(images):
    images_len = len(images)
    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    for i in range(0, images_len):
        ax = plt.subplot(2, 3, 1+i)
        #image = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        ax.imshow(images[i])
        ax.axis('off')
    # plt.subplots_adjust(left=0., right=1, top=0.7, bottom=0.)
    plt.show()


def find_lane(image):
    """尋找道路

    Args:
        image: (numpy.ndarray)影像
    Returns:
        reverse_p_transformation: (numpy.ndarray)輸出影像
        angle: (float)道路角度
        curve_radius: (numpy.float64)平均曲率半徑
        center_offset_pix: (int)偏移中心像素點
    """
    camera_calibrator = CameraCalibrator(IMAGE_SHAPE)
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
    # print(f"Vehicle is {abs(center_offset_pix):.3f}pixel {towards} of center.")

    # 逆透視轉換
    reverse_p_transformation = p_transformer.revert(sliding_window_search)

    cv2.imshow('reverse_p_transformation', reverse_p_transformation)
    #plot_images_2x3([image1_, image2])

    return reverse_p_transformation, angle, curve_radius, center_offset_pix


# =================== Main ==========================
if __name__ == "__main__":
    # 設置中斷
    signal.signal(signal.SIGINT, signal_handler)
    interrupted = False

    '''
    img = cv2.imread("./img/capture (11).jpg")
    # 校正鏡頭失真
    img1 = camera_calibrator.undistort_image(img)
    # cv2.imshow('img1', img1)
    image1 = img1.copy()

    image_lane, angle, _, _ = find_lane(image1)
    cv2.waitKey(0)
    '''

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP, TCP_PORT))
    s.listen(True)
    conn, addr = s.accept()
    print("連接位址: ", addr)

    intersection = Intersection()
    my_maze = Maze()
    my_aruco = Cv2Aruco()

    next_location = 0   # 下一次的地點
    last_location = 0   # 舊地點
    is_intersection = 'F'
    while True:
        # 接收Pi的值
        length = recvall(conn, 16)
        stringData = recvall(conn, int(length))

        next_location = int(str(stringData[:1], encoding="utf-8"))
        print(next_location)
        data = np.fromstring(stringData[2:], dtype='uint8')
        img = cv2.imdecode(data, 1)

        if next_location != last_location:
            start_point = my_maze.location_map[last_location]
            end_point = my_maze.location_map[next_location]
            start_dir = my_maze.now_dir
            _, intersection_turn = my_maze.walk_maze(start_point, end_point, start_dir)
            turn_num = 0

            last_location = next_location
            
            image_lane, angle, _, center_offset_pix = find_lane(img)

            if center_offset_pix > 20:
                angle = 100
            elif center_offset_pix < -20:
                angle = 80

            # 傳送值給Pi
            sock_str = str(is_intersection) + "," + str(f'{angle:.0f}')
            print(sock_str)
            conn.send(sock_str.encode('utf-8'))
            continue

        # 是否發現指定地點的 ArUco tag
        _, tag_id = my_aruco.find_aruco(img[-80:, :])
        if tag_id != 0:
            if tag_id == next_location:
                # 傳送值給Pi (停止)
                sock_str = str("E") + "," + str(0)
                conn.send(sock_str.encode('utf-8'))
        
        if next_location != 0:
            cv2.imshow('origin_img', img)
            # 判斷是否遇到路口
            is_intersection = 'T' if intersection.find_intersection(img) else 'F'
            if is_intersection == 'T':

                # intersection_type, _ = intersection.determine_intersection(img)
                turn_dir = intersection_turn[turn_num]
                turn_num += 1
                # turn_dir = {0:'other', 1:'right', 2:'left', 3:'forward'}
                # turn_dir = 2
                # 傳送值給Pi (轉彎)
                sock_str = str(is_intersection) + "," + str(turn_dir)
                print(sock_str)
                conn.send(sock_str.encode('utf-8'))

            else:
                image_lane, angle, _, center_offset_pix = find_lane(img)

                # if center_offset_pix > 30:
                #     angle = angle + abs((90 - angle) // 2)
                # elif center_offset_pix < -30:
                #     angle = angle - abs((90 - angle) // 2)

                if center_offset_pix > 20:
                    angle = 100
                elif center_offset_pix < -20:
                    angle = 80

                # 傳送值給Pi (尋車道)
                sock_str = str(is_intersection) + "," + str(f'{angle:.0f}')
                print(sock_str)
                conn.send(sock_str.encode('utf-8'))
        

        # ESC退出
        if cv2.waitKey(1) & 0xff == 27:
            break


    conn.close()
    s.close()
    cv2.destroyAllWindows()
