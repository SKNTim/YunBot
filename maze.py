import time
import math
import cv2
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from image_calibrator import CameraCalibrator
from image_calibrator import RoadMarkingDetector


class Cv2Aruco(object):
    """ OpenCV內的aruco tag """
    def __init__(self):
        self.place_name = { 1: "Pool", 2: "supermarket", 3: "School",
                            4: "Park", 5: "Library"}
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters_create()
        self.font = ImageFont.truetype('msjh.ttc', 25)
    
    def find_aruco(self, image):
        """在影像中尋找aruco tag

        Args:
            image: (numpy.ndarray)輸入影像
        Returns:
            image_markers: (numpy.ndarray)經過標記aruco tag的影像
            max_id: (int)最接近的aruco tag的ID
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict,
                                                    parameters=self.parameters)
        image_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)

        max_id = 0
        max_distance = 0
        if len(corners) > 0:
            for corner, id in zip(corners, ids):
                if id[0] in self.place_name:
                    x1 = corner[0][0]
                    x2 = corner[0][1]
                    d_x = x2[0] - x1[0]
                    d_y = x2[1] - x1[1]
                    distance = round(math.sqrt(d_x**2 + d_y**2))                
                    if distance > max_distance:
                        max_id = id[0]
                        max_distance = distance

                    # 在aruco tag上顯示地名
                    img_PIL = Image.fromarray(cv2.cvtColor(image_markers, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_PIL)
                    draw.text(tuple(x1), self.place_name[id[0]], font=self.font, fill=(255,0,255))
                    image_markers = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)

        return image_markers, max_id


class Maze(object):
    """ 迷宮 """
    maze_map = np.array([   [0,0,0,1,0,0,0],
                            [0,1,0,1,0,1,0],
                            [0,0,0,0,0,1,0],
                            [1,1,0,1,0,1,0],
                            [0,0,0,0,0,0,0]])
    location_map = {0:[4, 0], 1:[0, 1], 2:[1, 4], 3:[4, 5], 4:[2, 6], 5:[3, 2]}
    # {1:"加油站", 2:"火車站", 3:"學校", 4:"公園", 5:"圖書館"}
    
    def __init__(self, maze_map=maze_map, location_map=location_map):
        self.route_stack = [[0, 0]]   # 路線圖
        self.route_history = [[0, 0]]  # 走過的點
        self.maze_map = maze_map
        self.location_map = location_map
        self.maze_high = np.shape(self.maze_map)[0]
        self.maze_width = np.shape(self.maze_map)[1]
        self.now_point = [4, 0]
        self.now_dir = 1

    def calc_maze_turn_dir(self, start_dir, end_dir):
        """把點換算成轉彎方向

        Args:
            start_dir: (list-int)開始方向
            end_dir: (list-int)結束方向
        Returns:
            turn_dir: (int)轉彎方向
            # {0:other, 1:right, 2:left, 3:forward}
        """
        turn_dir = 0
        if start_dir[0] == 0:
            if end_dir[0] == 0:
                turn_dir = 3
            elif start_dir[1] * end_dir[0] > 0:
                turn_dir = 1
            elif start_dir[1] * end_dir[0] < 0:
                turn_dir = 2
        elif start_dir[1] == 0:
            if end_dir[1] == 0:
                turn_dir = 3
            elif start_dir[0] * end_dir[1] < 0:
                turn_dir = 1
            elif start_dir[0] * end_dir[1] > 0:
                turn_dir = 2

        return turn_dir

    def maze_flood_fill(self, start_point=[0, 0], end_point=[0, 4], start_dir=1):
        """迷宮，洪水演算法(flood fill)

        Args:
            maze: (numpy.ndarray)二維迷宮
            end_point: (numpy.ndarray)終點
        Returns:
            flood_maze: (numpy.ndarray)經過洪水演算法的迷宮
        """
        # 複製self.maze_map到flood_maze
        flood_maze = np.tile(self.maze_map, 1)
        # 把不能走的地方(1)設為-1
        flood_maze[flood_maze == 1] = -1
        # 判斷起點是否可以走
        if flood_maze[end_point[0], end_point[1]] == 0:
            # 將終點設為1
            flood_maze[end_point[0], end_point[1]] = 1
            num = 0
            while True:
                num += 1
                # flood_maze中等於num數字的位子
                num_points = np.where(flood_maze == num)
                if len(num_points[0]) == 0:
                    break
                for x, y in zip(num_points[0], num_points[1]):
                    if x != 0:
                        if flood_maze[x-1][y] == 0:
                            if start_dir == 4:
                                flood_maze[x-1][y] = num+1
                            if not ((x-1) == start_point[0] and y == start_point[1]):
                                flood_maze[x-1][y] = num+1
                    if x != (self.maze_high-1):
                        if flood_maze[x+1][y] == 0:
                            if start_dir == 3:
                                flood_maze[x+1][y] = num+1
                            if not ((x+1) == start_point[0] and y == start_point[1]):
                                flood_maze[x+1][y] = num+1
                    if y != 0:
                        if flood_maze[x][y-1] == 0:
                            if start_dir == 1:
                                flood_maze[x][y-1] = num+1
                            if not (x == start_point[0] and (y-1) == start_point[1]):
                                flood_maze[x][y-1] = num+1

                    if y != (self.maze_width-1):
                        if flood_maze[x][y+1] == 0:
                            if start_dir == 2:
                                flood_maze[x][y+1] = num+1
                            if not (x == start_point[0] and (y+1) == start_point[1]):
                                flood_maze[x][y+1] = num+1
        return flood_maze

    def walk_maze(self, start_point=[0, 0], end_point=[0, 4], start_dir=1):
        """走迷宮，並列出路線及遇岔路時轉彎方向

        Args:
            maze: (numpy.ndarray)二維迷宮
            start_point: (numpy.ndarray)起點
            end_point: (numpy.ndarray)終點
            start_dir: (int)起始方向 {1:right, 2:left, 3:up, 4:down}
        Returns:
            route_stack: (numpy.ndarray)路線
            intersection_turn: (numpy.ndarray)遇岔路時轉彎方向
        """
        maze2 = self.maze_flood_fill(start_point, end_point, start_dir)

        point_val = maze2[start_point[0], start_point[1]]

        is_have_road = False
        route_stack = []
        intersection_turn = []
        if point_val > 0:
            route_stack.append(start_point)

            point = route_stack[-1]
            if start_dir == 1:
                if point[1] != (self.maze_width-1):
                    if maze2[point[0]][point[1]+1] != -1:
                        maze2[point[0]][point[1]] = maze2[point[0]][point[1]+1] + 1
                        point_val = maze2[point[0]][point[1]+1]
                        route_stack.append([point[0], point[1]+1])
                        is_have_road = True
            elif start_dir == 2:
                if point[1] != 0:
                    if maze2[point[0]][point[1]-1] != -1:
                        maze2[point[0]][point[1]] = maze2[point[0]][point[1]-1] + 1
                        point_val = maze2[point[0]][point[1]-1]
                        route_stack.append([point[0], point[1]-1])
                        is_have_road = True
            elif start_dir == 3:
                if point[0] != 0:
                    if maze2[point[0]-1][point[1]] != -1:
                        maze2[point[0]][point[1]] = maze2[point[0]-1][point[1]] + 1
                        point_val = maze2[point[0]-1][point[1]]
                        route_stack.append([point[0]-1, point[1]])
                        is_have_road = True
            elif start_dir == 4:
                if point[0] != (self.maze_high-1):
                    if maze2[point[0]+1][point[1]] != -1:
                        maze2[point[0]][point[1]] = maze2[point[0]+1][point[1]] + 1
                        point_val = maze2[point[0]+1][point[1]]
                        route_stack.append([point[0]+1, point[1]])
                        is_have_road = True

            if is_have_road:
                for _ in range(point_val):
                    point_val -= 1
                    point = route_stack[-1]
                    if point[0] != 0:
                        if maze2[point[0]-1][point[1]] == point_val:
                            route_stack.append([point[0]-1, point[1]])
                            continue
                    if point[0] != (self.maze_high-1):
                        if maze2[point[0]+1][point[1]] == point_val:
                            route_stack.append([point[0]+1, point[1]])
                            continue
                    if point[1] != 0:
                        if maze2[point[0]][point[1]-1] == point_val:
                            route_stack.append([point[0], point[1]-1])
                            continue
                    if point[1] != (self.maze_width-1):
                        if maze2[point[0]][point[1]+1] == point_val:
                            route_stack.append([point[0], point[1]+1])
                            continue

                for index, route in enumerate(route_stack):
                    gap_num = 0
                    if route[0] != 0:
                        if self.maze_map[route[0]-1][route[1]] == 0:
                            gap_num += 1
                    if route[0] != (self.maze_high-1):
                        if self.maze_map[route[0]+1][route[1]] == 0:
                            gap_num += 1
                    if route[1] != 0:
                        if self.maze_map[route[0]][route[1]-1] == 0:
                            gap_num += 1
                    if route[1] != (self.maze_width-1):
                        if self.maze_map[route[0]][route[1]+1] == 0:
                            gap_num += 1

                    if gap_num > 2 and index != (len(route_stack)-1):
                        y1 = route_stack[index][0] - route_stack[index-1][0]
                        x1 = route_stack[index][1] - route_stack[index-1][1]
                        y2 = route_stack[index+1][0] - route_stack[index][0]
                        x2 = route_stack[index+1][1] - route_stack[index][1]
                        intersection_turn.append(self.calc_maze_turn_dir([y1, x1], [y2, x2]))
                
                x = route_stack[-2][0] - route_stack[-1][0]
                y = route_stack[-2][1] - route_stack[-1][1]
                if x == 0:
                    self.now_dir = 1 if y < 0 else 2
                else:
                    self.now_dir = 3 if x > 0 else 4
                self.now_point = end_point
            else:
                print("起始方向錯誤")
        else:
            print("無法到達迷宮終點")

        return route_stack, intersection_turn

    def maze_go(self, start_point=None, start_dir=None):
        """往前走到下一個地點或路口

        Args:
            start_point: (list_int)起始點
            start_dir: (int)起始方向
        """
        if start_point is None:
            start_point = self.now_point
        if start_dir is None:
            start_dir = self.now_dir

        flood_maze = np.tile(self.maze_map, 1)
        # 把不能走的地方(1)設為-1
        flood_maze[flood_maze == 1] = -1
        for location in list(self.location_map.values()):
            flood_maze[location[0], location[1]] = 1


        x = start_point[1]
        y = start_point[0]
        prev_dir = start_dir

        while True:
            # {1:right, 2:left, 3:up, 4:down}
            if start_dir == 1 and prev_dir != 2:
                if x != (self.maze_width-1):
                    if flood_maze[y, x+1] != -1:
                        start_dir = 1
                        prev_dir = 1
                        x += 1
                    else:
                        start_dir = 3
                else:
                    start_dir = 3
                
                
            elif start_dir == 2 and prev_dir != 1:
                if x != 0:
                    if flood_maze[y, x-1] != -1:
                        start_dir = 2
                        prev_dir = 2
                        x -= 1
                    else:
                        start_dir = 4
                else:
                    start_dir = 4
                
            elif start_dir == 3 and prev_dir != 4:
                if y != 0:
                    if flood_maze[y-1, x] != -1:
                        start_dir = 3
                        prev_dir = 3
                        y -= 1
                    else:
                        start_dir = 2
                else:
                    start_dir = 2

            elif start_dir == 4 and prev_dir != 3:
                if y != (self.maze_high-1):
                    if flood_maze[y+1, x] != -1:
                        start_dir = 4
                        prev_dir = 4
                        y += 1
                    else:
                        start_dir = 1
                else:
                    start_dir = 1
            
            else:
                print("maze_go Error")
                break
            
            # 假如有移動才開始判斷
            if not (x == start_point[1] and y == start_point[0]):
                if flood_maze[y, x] == 1:
                    print("location break")
                    break


                gap_num = 0
                if x != 0:
                    if flood_maze[y, x-1] != -1:
                        gap_num += 1
                if x != (self.maze_width-1):
                    if flood_maze[y, x+1] != -1:
                        gap_num += 1
                if y != 0:
                    if flood_maze[y-1, x] != -1:
                        gap_num += 1
                if y != (self.maze_high-1):
                    if flood_maze[y+1, x] != -1:
                        gap_num += 1
                
                if gap_num > 2:
                    print("intersection break")
                    break
            
        self.now_point = [y, x]
        self.now_dir = start_dir
        print("finish")

    def have_road(self, turn_dir=3):
        """想轉彎的方向是否有路

        Args:
            turn_dir: (string)轉彎方向
        Returns:
            _: (bool)是否有路
        """
        x = self.now_point[1]
        y = self.now_point[0]     

        # {1:right, 2:left, 3:up, 4:down}
        real_dir = self.calc_car_turn_dir(turn_dir)
        if real_dir == 1:
            if x != (self.maze_width-1):
                if self.maze_map[y, x+1] == 0:
                    return True
                else:
                    return False
            else:
                return False

        elif real_dir == 2:
            if x != 0:
                if self.maze_map[y, x-1] == 0:
                    return True
                else:
                    return False
            else:
                return False

        elif real_dir == 3:
            if y != 0:
                if self.maze_map[y-1, x] == 0:
                    return True
                else:
                    return False
            else:
                return False

        elif real_dir == 4:
            if y != (self.maze_high-1):
                if self.maze_map[y+1, x] == 0:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def calc_car_turn_dir(self, turn_dir=3):
        """更新車子轉彎後方向

        Args:
            turn_dir: (string)轉彎方向
            # {0:other, 1:right, 2:left, 3:forward}
        """
        if turn_dir == 3:
            real_dir = self.now_dir
        elif turn_dir == 1:
            if self.now_dir == 1: real_dir = 4
            if self.now_dir == 2: real_dir = 3
            if self.now_dir == 3: real_dir = 1
            if self.now_dir == 4: real_dir = 2
        elif turn_dir == 2:
            if self.now_dir == 1: real_dir = 3
            if self.now_dir == 2: real_dir = 4
            if self.now_dir == 3: real_dir = 2
            if self.now_dir == 4: real_dir = 1
        else:
            real_dir = 0

        return real_dir

    def in_intersection(self):
        """是否在路口

        Returns:
            _: (bool)是否在路口
        """
        x = self.now_point[1]
        y = self.now_point[0]
        # {1:right, 2:left, 3:up, 4:down}
        gap_num = 0
        if x != 0:
            if self.maze_map[y, x-1] == 0:
                gap_num += 1
        if x != (self.maze_width-1):
            if self.maze_map[y, x+1] == 0:
                gap_num += 1
        if y != 0:
            if self.maze_map[y-1, x] == 0:
                gap_num += 1
        if y != (self.maze_high-1):
            if self.maze_map[y+1, x] == 0:
                gap_num += 1

        if gap_num >= 3:
            return True
        else:
            return False


class Intersection(object):
    """ 路口 """
    def __init__(self):
        self.camera_calibrator = CameraCalibrator((320, 240))
        self.marking_detector = RoadMarkingDetector()

    def find_intersection(self, image):
        """是否有路口

        Args:
            image: (numpy.ndarray)影像
        Returns:
            is_intersection: (bool)是否有路口
        """
        img = image[-20:, :]
        # 對影像中的紅色做遮罩
        mask1 = self.marking_detector.find_hsv_mask(img, hsv_range="red1")
        mask2 = self.marking_detector.find_hsv_mask(img, hsv_range="red2")
        thresh = cv2.bitwise_or(mask1, mask2)

        if np.sum([thresh==255]) > 3000:
            is_intersection = True
        else:
            is_intersection = False
        return is_intersection

    def determine_intersection(self, image):
        """判斷路口類型

        Args:
            image: (numpy.ndarray)影像
        Returns:
            intersection_type: (list-int)路口類型[左, 前, 右]
            image: (numpy.ndarray)輸出影像
        """
        image_copy = image.copy()
        # 校正鏡頭失真
        image_copy = self.camera_calibrator.undistort_image(image_copy)
        image_copy = image_copy[:170, :]

        # 對影像中的紅色做遮罩
        mask1 = self.marking_detector.find_hsv_mask(image_copy, hsv_range="red1")
        mask2 = self.marking_detector.find_hsv_mask(image_copy, hsv_range="red2")
        thresh = cv2.bitwise_or(mask1, mask2)

        # 進行一次形態學侵蝕運算
        # 斷開色塊間連結
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.erode(thresh, kernel)

        # 尋找輪廓
        (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_copy, cnts, -1, (0, 255, 0), 1)

        # 剃除面積太小的輪廓
        cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > 150]

        # 計算各色塊中心點
        M = [cv2.moments(cnt) for cnt in cnts]
        midpoints = []
        for m in M:
            cX = int(m["m10"] / m["m00"])
            cY = int(m["m01"] / m["m00"])
            midpoints.append((cX, cY))
            cv2.circle(image_copy, (cX, cY), 5, (1, 227, 254), -1)

        # 判別路口類型，intersection_type = [左, 前, 右]
        intersection_type = [0, 0, 0]
        for midpoint in midpoints:
            if midpoint[0] < 120:
                intersection_type[0] = 1
            if midpoint[0] > 230:
                intersection_type[2] = 1
            if midpoint[1] < 100:
                intersection_type[1] = 1

        return intersection_type, image_copy


if __name__ == "__main__":
    my_maze = Maze()
    start_point = [0, 1]
    end_point = [2, 3]
    start_dir = 1

    stime = time.time()
    print(my_maze.maze_map)

    route_stack, intersection_turn = my_maze.walk_maze(start_point, end_point, start_dir)

    print(route_stack)
    print(intersection_turn)
    print(my_maze.now_dir)
    
    print(time.time()-stime)
