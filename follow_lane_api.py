import cv2
import time
import math
import sys
import signal
import socket
import numpy as np
from lanelines import LaneFinder
from image_calibrator import CameraCalibrator
from image_calibrator import PerspectiveTransform
from image_calibrator import RoadMarkingDetector
from maze import Maze
from maze import Cv2Aruco
from maze import Intersection


# TCP_IP = "192.168.137.1"    # 筆電開熱點
#TCP_IP = "192.168.0.155"   # 實驗室WIFI
TCP_IP = "172.20.10.4"      # 手機WIFI
# TCP_IP = "192.168.43.5"
TCP_PORT = 8004
IMAGE_SHAPE = (320, 240)


class Streaming(object):

    def __init__(self, tcp_address):
        # 選擇Socket類型和Socket數據包類型
        # AF_INET:於伺服器與伺服器之間進行串接
        # SOCK_STREAM:使用TCP(資料流)的方式提供可靠、雙向、串流的通信頻道
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 監聽的IP位址和Port
        self.s.bind(tcp_address)
        # 最多可接受多少socket串接
        self.s.listen(True)
        # 接收串接，並會回傳(client,address)串接對象與IP位址資訊
        self.conn, self.addr = self.s.accept()
        print("連接位址: ", self.addr)

    def __del__(self):
        print("Streaming物件已刪除")
        self.conn.close()
        self.s.close()

    def recvall(self, sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def sock_receive(self):
        """接收資料從Client端

        Returns:
            isLoop: (bool)是否繼續迴圈
            sock_list: (list)接收資料
            image: (numpy.ndarray)接收影像
        """
        sock_list = []
        image = ""
        isLoop = True
        try:
            length = self.recvall(self.conn, 16)
            bytes_data = self.recvall(self.conn, int(length))
            split_index = bytes_data.find(b';')
            
            data = str(bytes_data[:split_index], encoding="utf-8")
            sock_list = data.split(",")
            img_data = np.fromstring(bytes_data[split_index+1:], dtype='uint8')
            image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        except Exception as e:
            print("socket斷線: " + str(e))
            isLoop = False
        return isLoop, sock_list, image

    def sock_send(self, data=[]):
        """發送資料給Client端 "資料"

        Args:
            data: (list)發送資料
        Returns:
            isLoop: (bool)是否繼續迴圈
        """
        if len(data) > 0:
            sock_str = ",".join(str(v) for v in data)
        else:
            sock_str = "0"
        isLoop = True
        try:
            self.conn.send(sock_str.encode('utf-8'))
        except Exception as e:
            print("socket斷線: " + str(e))
            isLoop = False
        return isLoop


# 中斷
def signal_handler(signal, frame):
    cv2.destroyAllWindows()
    print('Stop pressing the CTRL+C!')
    sys.exit(0)


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

    # 逆透視轉換
    reverse_p_transformation = p_transformer.revert(sliding_window_search)
    cv2.imshow('reverse_p_transformation', reverse_p_transformation)

    return reverse_p_transformation, angle, curve_radius, center_offset_pix


# =================== Main ==========================
if __name__ == "__main__":
    # 設置中斷
    signal.signal(signal.SIGINT, signal_handler)

    # 宣告socket連線物件
    streaming = Streaming((TCP_IP, TCP_PORT))

    intersection = Intersection()
    my_maze = Maze()
    my_aruco = Cv2Aruco()

    next_location = 4   # 下一次的地點
    last_location = 0   # 舊地點

    while True:
        # 接收從Server端來的資料
        isLoop, data, image = streaming.sock_receive()
        action = data[0]

        print(f"===== {action} =====")
        sock_data = []
        # ------- send_image -------
        if action == "send_image":
            # 傳送影像
            # cv2.imshow('original_image', image)
            pass

        # ------- have_road -------
        elif action == "have_road":
            # 是否有路
            car_turn_dir = data[1]
            if car_turn_dir == "forward":
                car_turn_dir = 3
            elif car_turn_dir == "right":
                car_turn_dir = 1
            elif car_turn_dir == "left":
                car_turn_dir = 2
            
            # {1:right, 2:left, 3:up, 4:down}
            sock_data = [my_maze.have_road(car_turn_dir)]

        # ------- find_aruco -------
        elif action == "find_aruco":
            # AR_TAG是多少
            _, tag_id = my_aruco.find_aruco(image[-80:, :])
            sock_data = [tag_id]
        
        # ------- have_intersection -------
        elif action == "have_intersection":
            # 是否有路口
            sock_data = [intersection.find_intersection(image)]
        
        # ------- in_intersection -------
        elif action == "in_intersection":
            # 是否在路口
            sock_data = [my_maze.in_intersection()]
        
        # ------- follow_lane -------
        elif action == "follow_lane":
            # 依循道路
            image_lane, angle, _, center_offset_pix = find_lane(image)
            if center_offset_pix > 20:
                angle = 100
            elif center_offset_pix < -20:
                angle = 80
            else:
                angle = 90
            sock_data = [angle]
        
        # ------- maze_go -------
        elif action == "maze_go":
            # 更新車子位置(走到下個點)
            my_maze.maze_go()

        # ------- walk_maze -------
        elif action == "walk_maze":
            # 計算走迷宮的轉彎
            next_location = int(data[1])

            start_point = my_maze.location_map[last_location]
            end_point = my_maze.location_map[next_location]
            start_dir = my_maze.now_dir
            _, intersection_turn = my_maze.walk_maze(start_point, end_point, start_dir)
            last_location = next_location

            sock_data = intersection_turn
        
        # ------- update_car_dir -------
        elif action == "update_car_dir":
            # 更新車子轉彎後方向
            my_maze.now_dir = my_maze.calc_car_turn_dir(int(data[1]))
        
        # ------- other -------
        else:
            pass

        # =================================
        # 傳送結果資料到Client端
        isLoop = streaming.sock_send(sock_data)

        # ===== ESC退出 =====
        if cv2.waitKey(1) & 0xff == 27:
            break

    del streaming
    cv2.destroyAllWindows()
