# -*- coding: utf-8 -*-
import sys
import time
import signal
import socket
import numpy as np
import cv2
import serial
import speech_recognition as sr
from epaper import epd2in9
from snowboy import snowboydecoder

from cam_and_sound import PiCameraStream, Sound
from chatbot.chatbot import ChatBot


# 指定socket的IP與PORT
# TCP_IP = "192.168.137.1"    # 筆電開熱點
TCP_IP = "192.168.0.146"   # 實驗室WIFI
# TCP_IP = "192.168.43.5"
TCP_PORT = 8004

# snowboy model 位置路徑
MODEL = "snowboy/resources/models/yunbot.pmdl"
STT_TEXT = ""


class Streaming(object):

    def __init__(self, tcp_address):
        # 選擇Socket類型和Socket數據包類型
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 連線到指定的IP跟Port
        self.s.connect(tcp_address)
        self.data_list = []

    def __del__(self):
        print("Streaming物件已刪除")
        self.s.close()

    def sock_send_receive(self, data, image):
        """傳送、接收資料給Server端 "資料,影像"

        Args:
            data: (list)資料
            image: (numpy.ndarray)影像
        Returns:
            isLoop: (bool)是否繼續迴圈
            data_list: (list)接收資料
        """
        isLoop1 = self.sock_send(data, image)
        isLoop2, self.data_list = self.sock_receive()
        isLoop = isLoop1 and isLoop2
        return isLoop, self.data_list

    def sock_send(self, data=[], image=[]):
        """發送資料給Server端 "資料,影像"

        Args:
            data: (list)發送資料
            image: (numpy.ndarray)發送影像
        Returns:
            isLoop: (bool)是否繼續迴圈
        """
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, imgencode = cv2.imencode('.jpg', image, encode_param)
        img_data = np.array(imgencode)

        if len(data) > 0:
            data = ",".join(str(v) for v in data) + ";"
        else:
            data = ";"
        data = data.encode()
        stringData = data + img_data.tostring()
        isLoop = True
        try:
            self.s.send(str(len(stringData)).ljust(16).encode())
            self.s.send(stringData)
        except Exception as e:
            print("socket斷線: " + str(e))
            isLoop = False
        return isLoop

    def sock_receive(self):
        """接收資料從Server端

        Returns:
            isLoop: (bool)是否繼續迴圈
            sock_list: (list)接收資料
        """
        sock_list = []
        isLoop = True
        try:
            sock_str = self.s.recv(1024).decode('utf-8')
            sock_list = sock_str.split(',')
        except Exception as e:
            print("socket斷線: " + str(e))
            isLoop = False
        return isLoop, sock_list


# 中斷
def signal_handler(signal, frame):
    global interrupted
    interrupted = True
    move(1, "S", 10)
    # bot.save_unknown_input()
    cv2.destroyAllWindows()
    sound.pixel_ring.trace()
    print('Stop pressing the CTRL+C!')
    sys.exit(0)


def interrupt_callback():
    global interrupted
    return interrupted


def move(ID, action, parm):
    """組字串傳訊號給Arduino
    
    Args:
        ID: (int or string)編號
        action: (string)方向
        parm: (int)參數
    """
    ixyj = str(ID)+"_"+str(action)+"_"+str(parm)+'\n'
    ser.write(ixyj.encode('ascii'))
    ser.flush()
    
    if ID == 1:
        print(action, " --> ", parm)    
        if action == "F" or action == "B":
            delay = parm*64
        elif action == "R" or action == "L":
            delay = parm*9.2
        else:
            delay = parm
        time.sleep(delay/1000+0.5)


def txt_to_location(txt):
    """文字轉換為地點編號

    Args:
        txt: (string)字串
    Returns:
        location: (int)地點編號
    """
    if "gas station" in txt or "service station" in txt:
        location = 1
    elif "train station" in txt:
        location = 2
    elif "school" in txt:
        location = 3
    elif "park" in txt:
        location = 4
    elif "library" in txt:
        location = 5
    else:
        location = 0
        
    return location


def say_location(artag_id):
    if artag_id == 1:
        sound.word_to_sound("I arrived at the gas station.")
    elif artag_id == 2:
        sound.word_to_sound("I arrived at the train station.")
    elif artag_id == 3:
        sound.word_to_sound("I arrived at the school.")
    elif artag_id == 4:
        sound.word_to_sound("I arrived at the park.")
    elif artag_id == 5:
        sound.word_to_sound("I arrived at the library.")


def snowboy_callback():
    """ snowboy的callback，執行錄音 """
    global talk_txt
    detector.terminate()
    sound.record_wave()
    r = sr.Recognizer()
    with sr.AudioFile("record.wav") as source:
        sound_file = r.listen(source)
    talk_txt = ""
    try:
        # 語音辨識轉成文字
        print('Recognizing')
        talk_txt = r.recognize_google(sound_file)
        print(talk_txt)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("No response from Google Speech Recognition service: ", str(e))
    sound.pixel_ring.off()


# =================== Main ==========================
if __name__ == "__main__":    
    # 設置中斷
    signal.signal(signal.SIGINT, signal_handler)
    interrupted = False

    # 與Arduino通訊埠
    ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
    # snowboy初始設定
    detector = snowboydecoder.HotwordDetector(MODEL, sensitivity=0.5)

    # 宣告socket連線物件，指定IP跟Port
    streaming = Streaming((TCP_IP, TCP_PORT))
    # 啟動子執行緒
    picam = PiCameraStream(resolution=(320, 240), framerate=10).start()
    time.sleep(2)
    frame = picam.read()
    # rows, cols, ch = frame.shape
    print(np.shape(frame))

    sound = Sound()
    sound.pixel_ring.off()
    bot = ChatBot("Chatbot")
    bot.signon()

    state = "CHAT"
    talk_txt = ""
    next_location = 4   # 下一次的地點
    last_location = 0   # 舊地點
    isLoop = True
    # is_route = 'F'
    turn_num = 0    # 轉彎次數
    turn_dir = 0
    tLast = 0

    while isLoop:
        if time.time()-tLast > 0.01:
            tLast = time.time()
            frame = picam.read()

            print("===== " + state + " =====")
            # ------ CHAT ------
            if state == "CHAT":
                # 對話
                move(1, "S", 10)
                detector.start(detected_callback=snowboy_callback,
                               interrupt_check=interrupt_callback,
                               sleep_time=0.03)
                
                if talk_txt != "":
                    bot.get_input(talk_txt)
                    bot.respond()
                
                    if bot.action == "location":
                        # next_location = txt_to_location(bot.subject.strip().lower())
                        
                        txt_temp = bot.curr_response.strip().lower()
                        txt_index = txt_temp.find("the") + 4
                        next_location = txt_to_location(txt_temp[txt_index:])
                        
                        print("next_location = ", next_location)
                        if next_location == 0:
                            bot.curr_response = "No such location."
                            state = "CHAT"
                        else:

                            isLoop, data_list = streaming.sock_send_receive(["in_intersection"], frame)
                            if data_list[0] == "True":
                                state = "TURN"
                            else:
                                state = "ARTAG"

                            # 判斷地點是否更新
                            if next_location != last_location:
                                # 計算走迷宮的轉彎，data_list=[[intersection_turn]]
                                isLoop, data_list = streaming.sock_send_receive(["walk_maze", next_location], frame)
                                intersection_turn = [int(i) for i in data_list]
                                last_location = next_location
                                turn_num = 0
                            else:
                                say_location(artag_id)
                                state = "CHAT"
                                continue

                    elif bot.action == "move":
                        state = "DECIDE"
                    else:
                        state = "CHAT"
                
                    #if len(bot.curr_response) > 0:
                    #    sound.word_to_sound(bot.curr_response)

            # ------ DECIDE ------
            elif state == "DECIDE":
                # 如果要MOVE，先判斷是否有路可走
                direction = bot.curr_response.strip().lower()
                if direction.find("forward") != -1:
                    car_turn_dir = "forward"
                elif direction.find("right") != -1:
                    car_turn_dir = "right"
                elif direction.find("left") != -1:
                    car_turn_dir = "left"
                else:
                    print("I don't know")
                
                # 判斷此方向是否有路，data_list=[bool]
                isLoop, data_list = streaming.sock_send_receive(["have_road", car_turn_dir], frame)
                if data_list[0] == "True":
                    state = "MOVE"
                else:
                    state = "CHAT"
                    if car_turn_dir == "forward":
                        sound.word_to_sound("There is no road to go forward.")
                    elif car_turn_dir == "right":
                        sound.word_to_sound("There is no road to turn right.")
                    elif car_turn_dir == "left":
                        sound.word_to_sound("There is no road to turn left.")
            
            # ------ MOVE ------
            elif state == "MOVE":
                # 移動(前進到下一個地點或路口)
                direction = bot.curr_keyword.strip().lower()
                move(1, "S", 10)
                if car_turn_dir == "forward":
                    move(1, "S", 10)
                elif car_turn_dir == "right":
                    move(1, "R", 93)
                    # 更新車子轉彎後方向，data_list=[]
                    isLoop, data_list = streaming.sock_send_receive(["update_car_dir", 1], frame)
                elif car_turn_dir == "left":
                    move(1, "L", 93)
                    # 更新車子轉彎後方向，data_list=[]
                    isLoop, data_list = streaming.sock_send_receive(["update_car_dir", 2], frame)
                move(1, "F", 6)

                while True:
                    frame = picam.read()
                    # 判斷是否看到路口，data_list=[bool]
                    isLoop, data_list = streaming.sock_send_receive(["have_intersection"], frame)
                    have_intersection = (data_list[0] == "True")
                    if data_list[0] == "True":
                        # 有路口
                        move(1, "F", 18)
                        sound.word_to_sound("I am at an intersection. What should I do now?")
                        break

                    # 偵測AR Tag ID，data_list=[ID]
                    isLoop, data_list = streaming.sock_send_receive(["find_aruco"], frame)
                    artag_id = int(data_list[0])
                    if artag_id != 0:
                        # 有AR Tag
                        move(1, "F", 10)
                        say_location(artag_id)
                        break
        
                    # 依循道路，data_list=[angle]
                    isLoop, data_list = streaming.sock_send_receive(["follow_lane"], frame)
                    angle = int(data_list[0])
                    cx = 10 - round((angle-40)/10)
                    if cx > 10:
                        cx = 10
                    elif cx < 0:
                        cx = 0
                    print(cx)
                    move(2, str(cx), 1)
                
                # 更新車子位置(直走到下個點)
                isLoop, data_list = streaming.sock_send_receive(["maze_go"], frame)
                state = 'CHAT'

            # ------ ARTAG ------
            elif state == "ARTAG":
                # 找 ArUco Tag

                # 是否發現指定地點的 ArUco tag
                # 偵測AR Tag ID，data_list=[ID]
                isLoop, data_list = streaming.sock_send_receive(["find_aruco"], frame)
                artag_id = int(data_list[0])
                state = "INTERSECTION"
                if artag_id != 0:
                    if artag_id == next_location:
                        state = "CHAT"
                        move(1, "F", 8)
                        move(1, "S", 10)
                        say_location(artag_id)
                
            # ------ INTERSECTION ------
            elif state == "INTERSECTION":
                # 判斷是否看到路口，data_list=[bool]
                isLoop, data_list = streaming.sock_send_receive(["have_intersection"], frame)
                have_intersection = (data_list[0] == "True")
                if data_list[0] == "True":
                    state = "TURN"
                    move(1, "S", 10)
                    move(1, "F", 20)
                else:
                    state = "LANE"

            # ------ TURN ------
            elif state == "TURN":
                # 轉彎
                turn_dir = intersection_turn[turn_num]
                turn_num += 1
                # turn_dir = {0:'other', 1:'right', 2:'left', 3:'forward'}
                if turn_dir == 1:
                    move(1, "R", 93)
                elif turn_dir == 2:
                    move(1, "L", 93)
                elif turn_dir == 3:
                    pass
                else:
                    pass
                move(1, "F", 8)
                state = "LANE"

            # ------ LANE ------
            elif state == "LANE":
                # 依循道路，data_list=[angle]
                isLoop, data_list = streaming.sock_send_receive(["follow_lane"], frame)
                angle = int(data_list[0])
                cx = 10 - round((angle-40)/10)
                if cx > 10:
                    cx = 10
                elif cx < 0:
                    cx = 0
                print(cx)
                move(2, str(cx), 1)
                state = "ARTAG"

            # ------ else ------
            else:
                state = "CHAT"

        # data = ["AA", 3]
        # data_list = streaming.sock_send_receive(data, frame)
        # data_str = ",".join(str(v) for v in data_list)
        # print(data_str)

    del streaming
    del picam
    # bot.save_unknown_input()
    cv2.destroyAllWindows()
    ixyj = "1_S_5\n"
    ser.write(ixyj.encode('ascii'))
    ser.flush()
