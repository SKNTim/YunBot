#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import threading
from _thread import get_ident
import cv2
import numpy as np


class CameraEvent(object):
    """類似於事件的class，在新frame可用時向所有活動client端發出信號。
    """
    def __init__(self):
        self.events = {}

    def wait(self):
        ident = get_ident()
        if ident not in self.events:
            self.events[ident] = [threading.Event(), time.time()]
        return self.events[ident][0].wait()

    def set(self):
        now = time.time()
        remove = None
        for ident, event in self.events.items():
            # 此client端是否有設置event(事件)
            if not event[0].isSet():
                # 並將最後設置的時間戳更新為現在
                event[0].set()
                event[1] = now
            else:
                # client端沒有處理前一幀
                # 如果event(事件)保持設置超過5秒，
                # 則假設client端已經消失並將其刪除
                if now - event[1] > 5:
                    remove = ident
        
        if remove:
            del self.events[remove]

    def clear(self):
        self.events[get_ident()][0].clear()


class BaseCamera(object):
    thread = None  # 在背景線程讀取鏡頭frame
    frame = None  # 背景線程來的frame存儲在此
    last_access = 0  # 上次client端訪問鏡頭的時間
    event = CameraEvent()

    def __init__(self):
        if BaseCamera.thread is None:
            BaseCamera.last_access = time.time()

            # 開始背景線程
            BaseCamera.thread = threading.Thread(target=self._thread)
            BaseCamera.thread.start()

            while self.get_frame() is None:
                time.sleep(0)

    def get_frame(self):
        """Return the current camera frame."""
        BaseCamera.last_access = time.time()

        BaseCamera.event.wait()
        BaseCamera.event.clear()

        return BaseCamera.frame
    
    @staticmethod
    def frames():
        """Generator that returns frames from the camera."""
        raise RuntimeError('Must be implemented by subclasses.')

    @classmethod
    def _thread(cls):
        """Camera background thread."""
        print('Starting camera thread.')
        frames_iterator = cls.frames()
        for frame in frames_iterator:
            BaseCamera.frame = frame
            BaseCamera.event.set()  # 向client發送信號
            time.sleep(0)  # thread主動的讓出它執行cpu的所有權

            # 如果在過去10秒內沒有任何客戶要求幀，則停止該線程
            if time.time() - BaseCamera.last_access > 10:
                frames_iterator.close()
                print('Stopping camera thread due to inactivity.')
                break
        BaseCamera.thread = None
