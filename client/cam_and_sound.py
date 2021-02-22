# -*- coding: utf-8 -*-
import time
import json
import wave
import threading
import cv2
import serial
import pyaudio
import usb.core
import usb.util
import numpy as np
import speech_recognition as sr
from gtts import gTTS
from pygame import mixer
from epaper import epd2in9
from picamera import PiCamera
from picamera.array import PiRGBArray
from snowboy import snowboydecoder
from usb_pixel_ring_v2 import PixelRing
from PIL import Image, ImageDraw, ImageFont


#設定錄音參數值
CHUNK = 1024
FORMAT = pyaudio.paInt16
RATE = 16000
INDEX = None
CHANNELS = 1
RECORD_SECONDS = 3


# 接收攝影機串流影像，採用多執行緒的方式，降低緩衝區堆疊圖幀的問題。
class PiCameraStream(object):
    """Pi Camera 串流

    Args:
        resolution: (tuple)解析度(302,240)
        framerate: (int)幀率
    """
    def __init__(self, resolution=(320, 240), framerate=32):
        self.camera = PiCamera()
        # https://picamera.readthedocs.io/en/release-1.13/api_camera.html#picamera
        self.camera.resolution = resolution
        self.camera.framerate = framerate       # 幀率(0動態範圍幀率)
        # 測光模式('average', 'average'、'spot'、'backlit'、'matrix')
        # self.camera.meter_mode = 'average'
        # 白平衡('auto', 'off'、'auto'、'sunlight'、'cloudy'、'shade'、'tungsten'、'fluorescent'、'incandescent'、'flash'、'horizon')
        #self.camera.awb_mode = 'sunlight'
        # self.camera.saturation = 0    # 飽和度(0, -100~100)
        self.camera.brightness = 50   # 亮度(50, 0~100)
        # self.camera.iso = 0           # ISO(0自動, 0~1600)
        # self.camera.contrast = 0      # 對比度(0, -100~100)
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,
            format="bgr", use_video_port=True)
        
        self.frame = None
        self.stopped = False

    def __del__(self):
        print("PiCameraStream物件已刪除")
        self.stream.close()
        self.rawCapture.close()
        self.camera.close()

    def start(self):
        # 把程式放進子執行緒，daemon=True 表示該執行緒會隨著主執行緒關閉而關閉。
        print('ipcam started!')
        self.stopped = False
        threading.Thread(target=self.update, args=()).start()
        return self
 
    def update(self):
        for f in self.stream:
            self.frame = f.array
            self.rawCapture.truncate(0)
 
            if self.stopped:
                self.stream.close()
                self.rawCapture.close()
                self.camera.close()
                return

    def read(self):
        # 當有需要影像時，再回傳最新的影像。
        return self.frame
 
    def stop(self):
        self.stopped = True
        print('ipcam stopped!')

    def auto_awb(self, image):
        """自動白平衡(灰度世界算法)

        Args:
            image: (numpy.ndarray)影像
        Returns:
            auto_awb_image: (numpy.ndarray)自動白平衡影像
        """
        imgRGB = cv2.split(image)
        avgB = np.mean(imgRGB[0])
        avgG = np.mean(imgRGB[1])
        avgR = np.mean(imgRGB[2])
        avg = (avgB + avgG + avgR) / 3
        imgRGB[0] = np.minimum(imgRGB[0] * (avg / avgB), 255)
        imgRGB[1] = np.minimum(imgRGB[1] * (avg / avgG), 255)
        imgRGB[2] = np.minimum(imgRGB[2] * (avg / avgR), 255)
        auto_awb_image = cv2.merge([imgRGB[0], imgRGB[1], imgRGB[2]]).astype(np.uint8)
        return auto_awb_image


class Sound(object):
    """ 聲音相關，錄音、放音 """
    def __init__(self):
        # 抓取ReSpeaker 4 Mic Array 的 INDEX
        pa = pyaudio.PyAudio()
        for i in range(pa.get_device_count()):
            dev = pa.get_device_info_by_index(i)
            if "ReSpeaker 4 Mic Array" in dev['name']:
                INDEX = dev['index']

        if INDEX is None:
            print("pyaudio device no found!")

        # 文字轉語音存的檔名
        self.sound_file_name = 'wordToSound.mp3'
        self.record_file_name = "record.wav"
        
        # ReSpeaker 4 Mic Array LED USB設定
        ring_dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
        self.pixel_ring = PixelRing(ring_dev)

    def record_wave(self, record_sec=4):
        """錄音

        Args:
            record_sec: (int)錄音秒數
        """
        # 設定錄音參數值
        chunk = 1024
        rate = 16000
        channels = 1

        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=channels,
                         input_device_index=INDEX, rate=rate,
                         input=True, frames_per_buffer=chunk)
        self.pixel_ring.listen()
        '''
        while True:
            buffer = []
            for _ in range(0, int(rate/chunk*1)):
                data = stream.read(chunk)
                buffer.append(data)
            audio_data = np.fromstring(data, dtype=np.short)
            print(str(np.max(audio_data)))
            if np.max(audio_data) > 550:
                break
        
        temp_txt = 'Recording'
        print(temp_txt)
        second = 0
        while True:
            # temp_txt = temp_txt + '.'
            # print(temp_txt, end="\r", flush=True)
            for _ in range(0, int(rate/chunk*2)):
                data = stream.read(chunk)
                buffer.append(data)
            audio_data = np.fromstring(data, dtype=np.short)
            print(str(np.max(audio_data)))
            second += 1
            if np.max(audio_data) < 500 or second > 5:
                break
        '''
        print('Recording...')
        buffer = []
        for _ in range(0, int(rate/chunk*record_sec)):
            data = stream.read(chunk)
            buffer.append(data)
        
        print('Record Done\n')
        self.pixel_ring.think()
        stream.stop_stream()
        stream.close()
        pa.terminate()
        wf = wave.open(self.record_file_name, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(buffer))
        wf.close()

    def word_to_sound(self, text):
        """文字轉語音播放

        Args:
            text: (string)文字
        """
        tts = gTTS(text)
        tts.save(self.sound_file_name)
        self.play_sound(self.sound_file_name)

    def play_sound(self, file_name):
        """播放錄音

        Args:
            file_name: (string)要播放音檔名
        """
        self.pixel_ring.speak()
        mixer.init()
        mixer.music.load(file_name)
        mixer.music.play()
        while mixer.music.get_busy() == True:
            continue
        mixer.music.stop()
        mixer.quit()
        self.pixel_ring.off()
