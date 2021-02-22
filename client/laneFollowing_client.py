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


# 指定socket的IP與PORT
TCP_IP = "192.168.0.192"
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
    cv2.destroyAllWindows()
    picam.stop()
    print('Stop pressing the CTRL+C!')
    move(1, "S", 10)
    sound.pixel_ring.trace()
    sys.exit(0)

def interrupt_callback():
    global interrupted
    return interrupted
    

# 組字串傳訊號給Arduino
def move(ID, action, parm):
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
    global talk_txt
    global next_location
    
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

    reply = ""
    parm = ""
    action = ""
    for conversation in sound.conversations:
        for problem in conversation["problem"]:

            pattern = re.compile(problem, re.I)
            match = pattern.search(talk_txt)
                
            if match:
                print(match.group())
                if "?P" in problem:
                    parm = match.group(1)
                    print(parm)

                reply = random.choice(conversation["reply"])
                action = conversation["action"]
                break
        else:
            continue
        break


    if action == "move":
        reply += ", go to " + parm

    if reply == "":
        sound.word_to_sound("I can't understand what you said.")
    else:
        sound.word_to_sound(reply)

    if action == "move":
        next_location = txt_to_location(parm)
    else:
        detector.start(detected_callback=snowboy_callback,
                interrupt_check=interrupt_callback,
                sleep_time=0.03)
        detector.terminate()

        
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
sock.connect((TCP_IP, TCP_PORT))
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]


# 啟動子執行緒
picam = PiCameraStream(resolution=(320,240), framerate=10).start()
time.sleep(2)
frame = picam.read()
rows, cols, ch = frame.shape
print(np.shape(frame))

sound = Sound()

state = "N"
talk_txt = ""
next_location = 4
isLoop = True
is_route = 'F'
turn_dir = 0
tLast = 0

detector.start(detected_callback=snowboy_callback,
                interrupt_check=interrupt_callback,
                sleep_time=0.03)
print("next_location = ", next_location)


while isLoop:
    if time.time()-tLast > 0.01:
        tLast = time.time()
        frame = picam.read()
        result, imgencode = cv2.imencode('.jpg', frame, encode_param)
        data = np.array(imgencode)
        stringData = (str(next_location) + ",").encode() + data.tostring()
        
        
        try:
            sock.send( str(len(stringData)).ljust(16).encode() )
            sock.send( stringData )

            sock_str = sock.recv(1024).decode('utf-8')
            sock_list = sock_str.split(',')
            is_route = sock_list[0]
            
        except:
            print("socket斷線")
            isLoop = False
            break
        

        if is_route == 'E':
            move(1, "F", 10)
            move(1, "S", 500)
            # 等待指令
            detector.start(detected_callback=snowboy_callback,
                            interrupt_check=interrupt_callback,
                            sleep_time=0.03)
            print("next_location = ", next_location)
            
            
        elif is_route == 'T':
            turn_dir = int(sock_list[1])
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

        else:
            angle = int(sock_list[1])
            
            cx = 10 - round((angle-40)/10)
            if cx > 10:
                cx = 10
            elif cx < 0:
                cx = 0
            
            print(cx)
            move(2, str(cx), 1)
        

sock.close()
cv2.destroyAllWindows()
picam.stop()
ixyj = "1_S_5\n"
ser.write(ixyj.encode('ascii'))
ser.flush()

