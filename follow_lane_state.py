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


TCP_IP = "192.168.0.146"
# TCP_IP = "192.168.0.192"
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


def sock_receive():
    """接收Pi的值

    Returns:
        car_state: (string)車子狀態
        next_location: (int)下一個地點
        img: (numpy.ndarray)影像
    """
    length = recvall(conn, 16)
    stringData = recvall(conn, int(length))

    car_state = str(stringData[:2], encoding="utf-8")
    next_location = int(str(stringData[3:4], encoding="utf-8"))
    print(f"next_location = {next_location}")
    data = np.fromstring(stringData[5:], dtype='uint8')
    img = cv2.imdecode(data, 1)

    return car_state, next_location, img


def sock_send(state, sock_data):
    """傳送值給Pi "狀態,資料"

    Args:
        state: (string)狀態
        sock_data: (string or int)資料
    """
    # 傳送值給Pi "狀態,資料"
    sock_str = str(state[:2]) + "," + str(sock_data)
    print(sock_str)
    conn.send(sock_str.encode('utf-8'))


def plot_images_2x3(images):
    images_len = len(images)
    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    for i in range(0, images_len):
        ax = plt.subplot(2, 3, 1+i)
        # image = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
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
    # plot_images_2x3([image1_, image2])

    return reverse_p_transformation, angle, curve_radius, center_offset_pix


# =================== Main ==========================
if __name__ == "__main__":
    # 設置中斷
    signal.signal(signal.SIGINT, signal_handler)
    interrupted = False

    # 選擇Socket類型和Socket數據包類型
    # AF_INET:於伺服器與伺服器之間進行串接
    # SOCK_STREAM:使用TCP(資料流)的方式提供可靠、雙向、串流的通信頻道
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 監聽的IP位址和Port
    s.bind((TCP_IP, TCP_PORT))
    # 最多可接受多少socket串接
    s.listen(True)
    # 接收串接，並會回傳(client,address)串接對象與IP位址資訊
    conn, addr = s.accept()
    print("連接位址: ", addr)

    intersection = Intersection()
    my_maze = Maze()
    my_aruco = Cv2Aruco()

    next_location = 0   # 下一次的地點
    last_location = 0   # 舊地點

    state = "CHAT"
    while True:
        car_state, next_location, img = sock_receive()
        print(f"===== {state} =====")

        sock_data = 0
        # ------ CHAT ------
        if state == "CHAT":
            # 對話
            if car_state == 'CH':
                state = "CHAT"
            elif car_state == 'DE':
                state = "DECIDE"
            elif car_state == 'AR':
                state = "ARTAG"

        # ------ decide ------
        elif state == "DECIDE":
            # 判斷此方向是否有路
            print("DECIDE")
            sock_send(state, sock_data)
            car_state, next_location, img = sock_receive()
            if my_maze.have_road(car_state):
                state = "MOVE"
            else:
                state = "CHAT"
            
            sock_send(state, sock_data)
            car_state, next_location, img = sock_receive()

        # ------ MOVE ------
        elif state == "MOVE":
            # 移動(前進到下一個地點或路口)\
            # 傳送值給Pi "狀態,資料"
            sock_send(state, sock_data)
            while True:
                car_state, next_location, img = sock_receive()
                # 是否發現地點(ArUco tag)或路口
                _, tag_id = my_aruco.find_aruco(img[-80:, :])
                if tag_id != 0 or intersection.find_intersection(img):
                    # 更新地點
                    my_maze.maze_go()
                    state = "CHAT"
                    break
                # 依循車道
                image_lane, angle, _, center_offset_pix = find_lane(img)
                if center_offset_pix > 20:
                    angle = 100
                elif center_offset_pix < -20:
                    angle = 80
                
                sock_data = str(f'{angle:.0f}')

                # 傳送值給Pi "狀態,資料"
                sock_send(state, sock_data)

        # ------ ARTAG ------
        elif state == "ARTAG":
            # 找 ArUco Tag
            if next_location != last_location:
                start_point = my_maze.location_map[last_location]
                end_point = my_maze.location_map[next_location]
                start_dir = my_maze.now_dir
                _, intersection_turn = my_maze.walk_maze(start_point, end_point, start_dir)
                turn_num = 0
                last_location = next_location

            # 是否發現指定地點的 ArUco tag
            _, tag_id = my_aruco.find_aruco(img[-80:, :])
            state = "INTERSECTION"
            if tag_id != 0:
                my_maze.now_point = my_maze.location_map.get(tag_id, my_maze.now_point)
                if tag_id == next_location:
                    state = "CHAT"

        # ------ INTERSECTION ------
        elif state == "INTERSECTION":
            # 判斷是否遇到路口
            if intersection.find_intersection(img):
                state = "TURN"
            else:
                state = "LANE"

        # ------ TURN ------
        elif state == "TURN":
            # 轉彎
            sock_data = intersection_turn[turn_num]
            turn_num += 1
            #if car_state == 'LA':
            state = "LANE"

        # ------ LANE ------
        elif state == "LANE":
            # 依循車道
            image_lane, angle, _, center_offset_pix = find_lane(img)
            if center_offset_pix > 20:
                angle = 100
            elif center_offset_pix < -20:
                angle = 80
            
            if car_state == 'AR':
                state = "ARTAG"
            sock_data = str(f'{angle:.0f}')

        # ------ else ------
        else:
            state = "CHAT"

        # 傳送值給Pi "狀態,資料"
        sock_send(state, sock_data)

        # ===== ESC退出 =====
        if cv2.waitKey(1) & 0xff == 27:
            break


    conn.close()
    s.close()
    cv2.destroyAllWindows()
