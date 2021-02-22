# -*- coding: utf-8 -*-
import re
import sys
import time
import math
import random
import signal
import socket
import threading
import numpy as np
import cv2
import serial
import wave
import speech_recognition as sr
from epaper import epd2in9
from snowboy import snowboydecoder
from PIL import Image, ImageDraw, ImageFont

from cam_and_sound import PiCameraStream, Sound
from chatbot.chatbot import ChatBot


# 指定socket的IP與PORT
TCP_IP = "192.168.0.146"
# TCP_IP = "192.168.43.5"
TCP_PORT = 8003

# snowboy model 位置路徑
MODEL = "snowboy/resources/models/yunbot.pmdl"
STT_TEXT = ""


# 中斷
def signal_handler(signal, frame):
    global interrupted
    interrupted = True
    sock.close()
    bot.save_unknown_input()
    cv2.destroyAllWindows()
    picam.stop()
    print('Stop pressing the CTRL+C!')
    move(1, "S", 10)
    sound.pixel_ring.trace()
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


def sock_send_receive(state, next_location, frame):
    """傳送、接收值給Pi "狀態,地點,影像"

    Args:
        state: (string)狀態
        next_location: (string)下一個地點
        frame: (numpy.ndarray)影像
    Returns:
        isLoop: (bool)是否繼續迴圈
        computer_state: (string)電腦狀態
        sock_data: (int)資料數值
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, imgencode = cv2.imencode('.jpg', frame, encode_param)
    data = np.array(imgencode)
    stringData = ((str(state) + ",").encode() +
                  (str(next_location) + ",").encode() +
                  data.tostring())
    isLoop = True
    computer_state = ""
    sock_data = ""
    try:
        sock.send(str(len(stringData)).ljust(16).encode())
        sock.send(stringData)

        sock_str = sock.recv(1024).decode('utf-8')
        sock_list = sock_str.split(',')
        computer_state = sock_list[0]
        sock_data = int(sock_list[1])
    except Exception as e:
        print("socket斷線")
        print(e)
        isLoop = False
    return isLoop, computer_state, sock_data


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
        #語音辨識轉成文字
        print('Recognizing')
        talk_txt = r.recognize_google(sound_file)
        print(talk_txt)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("No response from Google Speech Recognition service: {0}".format(e))
    sound.pixel_ring.off()

        
#################################################
# 設置中斷
signal.signal(signal.SIGINT, signal_handler)
interrupted = False

# 與Arduino通訊埠
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
# snowboy初始設定
detector = snowboydecoder.HotwordDetector(MODEL, sensitivity=0.5)

# 選擇Socket類型和Socket數據包類型
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 指定要串接的IP位址跟Port號
sock.connect((TCP_IP, TCP_PORT))

# 啟動子執行緒
picam = PiCameraStream(resolution=(320,240), framerate=10).start()
time.sleep(2)
frame = picam.read()
rows, cols, ch = frame.shape
print(np.shape(frame))

sound = Sound()
bot = ChatBot("Chatbot")
bot.signon()

state = "CH"
talk_txt = ""
next_location = 4
isLoop = True
is_route = 'F'
turn_dir = 0
tLast = 0

while isLoop:
    if time.time()-tLast > 0.01:
        tLast = time.time()
        frame = picam.read()
        isLoop, computer_state, sock_data = sock_send_receive(state, next_location, frame)
        if not isLoop:
            break

        print("===== " + state + " =====")
        # ------ CHAT ------
        if state == "CH":
            # 對話
            move(1, "S", 500)
            detector.start(detected_callback=snowboy_callback,
                            interrupt_check=interrupt_callback,
                            sleep_time=0.03)

            bot.get_input(talk_txt)
            bot.respond()
            if bot.action == "location":
                state = "AR"
                next_location = txt_to_location(bot.subject.strip().lower())
                print("next_location = ", next_location)
            elif bot.action == "move":
                state = "DE"
            else:
                state = "CH"
            
            if len(bot.curr_response) > 0:
                sound.word_to_sound(bot.curr_response)

        # ------ decide ------
        elif state == "DE":
            isLoop, computer_state, sock_data = sock_send_receive(state, next_location, frame)
            direction = bot.curr_keyword.strip().lower()
            if direction.find("forward") != -1:
                state = "MF"
            elif direction.find("right") != -1:
                state = "MR"
            elif direction.find("left") != -1:
                state = "ML"
            else:
                print("I don't know")
                
            isLoop, computer_state, sock_data = sock_send_receive(state, next_location, frame)
            state = computer_state
            if state != "MO":                
                print("There is no road over there.")
                sound.word_to_sound("There is no road over there.")

        
        # ------ MOVE ------
        elif state == "MO":
            # 移動(前進到下一個地點或路口)
            direction = bot.curr_keyword.strip().lower()
            move(1, "S", 10)
            if direction.find("forward") != -1:
                move(1, "S", 10)
            elif direction.find("right") != -1:
                move(1, "R", 93)
            elif direction.find("left") != -1:
                move(1, "L", 93)
            move(1, "F", 6)

            while True:
                frame = picam.read()
                isLoop, computer_state, sock_data = sock_send_receive(state, next_location, frame)
                if not isLoop:
                    break
                if computer_state == "CH":
                    move(1, "F", 20)
                    move(1, "S", 10)
                    break

                angle = sock_data
                cx = 10 - round((angle-40)/10)
                if cx > 10:
                    cx = 10
                elif cx < 0:
                    cx = 0
                print(cx)
                move(2, str(cx), 1)
            
            
            state = 'CH'

        # ------ ARTAG ------
        elif state == "AR":
            # 找 ArUco Tag
            state = computer_state
            
        # ------ INTERSECTION ------
        elif state == "IN":
            # 判斷是否遇到路口
            state = computer_state

        # ------ TURN ------
        elif state == "TU":
            # 轉彎
            turn_dir = sock_data
            # turn_dir = {0:'other', 1:'right', 2:'left', 3:'forward'}
            move(1, "S", 10)
            move(1, "F", 20)
            if turn_dir == 1:
                move(1, "R", 93)
            elif turn_dir == 2:
                move(1, "L", 93)
            elif turn_dir == 3:
                pass
            else:
                pass
            move(1, "F", 6)
            state = "LA"

        # ------ LANE ------
        elif state == "LA":
            # 依循車道
            angle = sock_data            
            cx = 10 - round((angle-40)/10)
            if cx > 10:
                cx = 10
            elif cx < 0:
                cx = 0            
            print(cx)
            move(2, str(cx), 1)

            state = "AR"

        # ------ else ------
        else:
            state = "CH"


sock.close()
bot.save_unknown_input()
cv2.destroyAllWindows()
picam.stop()
ixyj = "1_S_5\n"
ser.write(ixyj.encode('ascii'))
ser.flush()
