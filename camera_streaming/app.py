#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import math
import json
import socket
from importlib import import_module

import cv2
import numpy as np
from cv2 import aruco
from PIL import Image, ImageDraw, ImageFont

from camera_calibrator import PerspectiveTransform
# from camera_opencv import Camera
from camera_pi import Camera
from flask_bootstrap import Bootstrap
from flask import Flask, Response, render_template


app = Flask(__name__)
bootstrap = Bootstrap(app)

ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_50)
PARAMETERS = aruco.DetectorParameters_create()


class ImageProcessing(Camera):
    def __init__(self):
        super().__init__()
        self.camera_size = (640, 480)
        # self.src_vector = [[65,0],[288,0],[318,230],[43,230]]
        self.src_vector = [[130,5],[576,5],[636,460],[86,460]]
        self.p_transformer = PerspectiveTransform(self.camera_size, src_vector=self.src_vector)
        self.tag_corners = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
        self.old_tag_corners = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
        self.center = (0, 0)

        path_a = np.array([[i, 415] for i in range(30, 586)])
        path_b = np.array([[585, i] for i in range(55, 416)])
        path_c = np.array([[i, 55] for i in range(405, 586)])
        path_d = np.array([[405, i] for i in range(55, 416)])
        path_e = np.array([[i, 415] for i in range(240, 406)])
        path_f = np.array([[240, i] for i in range(55, 416)])
        path_g = np.array([[i, 55] for i in range(60, 241)])
        path_h = np.array([[60, i] for i in range(55, 236)])
        path_i = np.array([[i, 235] for i in range(60, 406)])
        path_a = np.vstack((path_a, path_b))
        path_a = np.vstack((path_a, path_c))
        path_a = np.vstack((path_a, path_d))
        path_a = np.vstack((path_a, path_e))
        path_a = np.vstack((path_a, path_f))
        path_a = np.vstack((path_a, path_g))
        path_a = np.vstack((path_a, path_h))
        path_a = np.vstack((path_a, path_i))
        self.path = path_a

    def get_frame_aruco(self):
        """取得frame，並經過找aruco、透視轉換、上下顛倒

        Returns:
            image_out: (numpy.ndarray)校正後影像
            tag_point: (tuple)aruco tag座標
            image_markers: (numpy.ndarray)標記aruco後影像(未校正影像)
        """
        frame = self.get_frame()
        img_data = np.fromstring(frame, dtype='uint8')
        image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

        self.tag_corners, image_markers = self.find_aruco_num(image, 49)
        if len(self.tag_corners) > 0:
            self.center = [int(self.tag_corners[:, 0].mean()),
                           int(self.tag_corners[:, 1].mean())]

        image = self.p_transformer.transform(image)
        M = cv2.getRotationMatrix2D((self.camera_size[0]/2, self.camera_size[1]/2),
                                    180, 1.0)
        image = cv2.warpAffine(image, M, self.camera_size)

        image_out = cv2.imencode('.jpg', image)[1].tobytes()
        image_markers = cv2.imencode('.jpg', image_markers)[1].tobytes()

        return image_out, image_markers

    def find_aruco_num(self, image, find_id, aruco_dict=None):
        """找尋指定aruco tag

        Args:
            image: (numpy.ndarray)輸入影像
            find_id: (int)欲找尋aruco tag id
        Returns:
            tag_corners: (tuple)aruco tag座標
            image_markers: (numpy.ndarray)標記aruco後影像
        """
        if aruco_dict==None:
            aruco_dict = ARUCO_DICT
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=PARAMETERS)
        image_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)

        tag_corners = np.array([])
        if len(corners) > 0 and find_id in ids:
            result = np.where(ids == find_id)
            c = corners[result[0][0]][0]

            tag_corners = np.array([self.transformer_point(point)
                                    for point in c])

        return tag_corners, image_markers

    def map_add_point(self, image, tag_corners):
        """加入aruco tag標記

        Args:
            image: (numpy.ndarray)輸入影像
            tag_corners: (tuple)aruco tag座標
        Returns:
            image_out: (numpy.ndarray)標記aruco後影像
        """
        img_data = np.fromstring(image, dtype='uint8')
        image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

        image_map = cv2.imread('map.png')

        if len(tag_corners) > 0:
            self.old_tag_corners = tag_corners
            image_map = self.draw_arrow(image_map, tag_corners, (255, 255, 0))
        else:
            image_map = self.draw_arrow(image_map, self.old_tag_corners, (220, 220, 220))
            
        image_out = cv2.imencode('.jpg', image_map)[1].tobytes()
        return image_out

    def transformer_point(self, point):
        point_vector = np.dot(self.p_transformer.transform_matrix,
                              np.array([[point[0]], [point[1]], [1]]))
        out_point = [self.camera_size[0]-int(point_vector[0]/point_vector[2]),
                     self.camera_size[1]-int(point_vector[1]/point_vector[2])]
        return out_point

    def calc_point(self, center, front):
        x = front[0] - center[0]
        y = front[1] - center[1]
        angle = math.atan2(y, x)
        distance = math.sqrt(math.pow(x, 2) + math.pow(y, 2))
        extend = [0, 0]
        extend[0] = int(math.cos(angle) * distance * 2 + center[0])
        extend[1] = int(math.sin(angle) * distance * 2 + center[1])
        return extend

    def draw_arrow(self, image, tag_corners, color=(255, 255, 0)):
        center = [int(tag_corners[:, 0].mean()), int(tag_corners[:, 1].mean())]
        tag_corners_scale = np.array([self.calc_point(center, corner)
                                      for corner in tag_corners])

        arrow1 = np.array([int(tag_corners_scale[:2, 0].mean()), int(tag_corners_scale[:2, 1].mean())])
        arrow2 = tag_corners_scale[3]
        temp = np.array([int(tag_corners_scale[2:, 0].mean()), int(tag_corners_scale[2:, 1].mean())])
        arrow3 = np.rint((temp - arrow1) * 0.7) + arrow1
        arrow4 = tag_corners_scale[2]
        
        arrow = np.array([arrow1, arrow2, arrow3, arrow4], np.int32)
        arrow = arrow.reshape((-1, 1, 2))
        cv2.polylines(image, [arrow], True, color, 3)

        return image

    def move_nearest_point(self):
        tag_corners_out = self.tag_corners.copy()
        distance_short = 1000
        point_short = np.array([0, 0])
        # center = [int(tag_corners_out[:, 0].mean()), int(tag_corners_out[:, 1].mean())]
        
        for point in self.path:
            # distance = np.sqrt(np.sum(np.square(point - self.center)))
            distance = np.sum(np.abs(point - self.center))
            if distance < distance_short:
                distance_short = distance
                point_short = point

        point_distance = point_short - self.center
        tag_corners_out += point_distance

        return tag_corners_out, distance_short


camera = ImageProcessing()


@app.route('/')
def index():
    return render_template('index.html', host_ip=get_host_ip())


@app.route('/a')
def a():
    return render_template('a.html')


@app.route('/point', methods=['GET', 'POST'])
def point():
    point = list(camera.center)
    point_axis = [0, 0]
    x_axis = [105, 195, 285, 360, 450, 540, 640]
    y_axis = [100, 190, 280, 370, 480]
    # x_axis = [52, 97, 142, 180, 225, 270, 320]
    # y_axis = [50, 95, 140, 185, 240]
    
    for index, x in enumerate(x_axis):
        if point[0] < x:
            point_axis[0] = index
            break

    for index, y in enumerate(y_axis):
        if point[1] < y:
            point_axis[1] = index
            break
    
    pointInfo = {}
    pointInfo['center'] = camera.center
    pointInfo['point_axis'] = point_axis
    return json.dumps(pointInfo)


def gen():
    """串流影像生成器(Generators)函數"""
    while True:
        frame, _ = camera.get_frame_aruco()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               frame + b'\r\n')


def gen2():
    """串流影像生成器(Generators)函數"""
    while True:
        image_out, _ = camera.get_frame_aruco()
        tag_corners_out = []
        if len(camera.tag_corners) > 0:
            tag_corners_out, _ = camera.move_nearest_point()
        frame = camera.map_add_point(image_out, tag_corners_out)
        
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               frame + b'\r\n')


def gen3():
    """串流影像生成器(Generators)函數"""
    while True:
        frame = camera.get_frame()
        # image_out, _ = camera.get_frame_aruco()
        
        '''
        frame = camera.get_frame()
        img_data = np.fromstring(frame, dtype='uint8')
        image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        tag_corners1, _ = camera.find_aruco_num(image, 49, aruco_dict=aruco_dict)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)
        tag_corners2, _ = camera.find_aruco_num(image, 49, aruco_dict=aruco_dict)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        tag_corners3, _ = camera.find_aruco_num(image, 49, aruco_dict=aruco_dict)

        image = camera.p_transformer.transform(image)
        M = cv2.getRotationMatrix2D((camera.camera_size[0]/2, camera.camera_size[1]/2),
                                    180, 1.0)
        image = cv2.warpAffine(image, M, camera.camera_size)

        if len(tag_corners1) > 0:
            center1 = [int(tag_corners1[:, 0].mean()), int(tag_corners1[:, 1].mean())]
            cv2.circle(image, tuple(center1), 4, (255, 0, 255), 3)
        if len(tag_corners2) > 0:
            center2 = [int(tag_corners2[:, 0].mean()), int(tag_corners2[:, 1].mean())]
            cv2.circle(image, tuple(center2), 4, (255, 0, 255), 3)
        if len(tag_corners3) > 0:
            center3 = [int(tag_corners3[:, 0].mean()), int(tag_corners3[:, 1].mean())]
            cv2.circle(image, tuple(center3), 4, (255, 0, 255), 3)
        
        frame = cv2.imencode('.jpg', image)[1].tobytes()
        '''
        
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               frame + b'\r\n')
        

@app.route('/video_feed')
def video_feed():
    """串流影像的路由(route)"""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/steam_map')
def steam_map():
    return Response(gen2(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/steam_test')
def steam_test():
    return Response(gen3(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def get_host_ip():
    """取得IP位址"""
    ip = ""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


if __name__ == '__main__':
    app.run(host=get_host_ip(), port=5000, threaded=True)
