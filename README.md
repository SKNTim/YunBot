# RBB Car 依循車道

### 檔案介紹

    電腦端
    |-- camera_streaming (觀測整個地圖的server)
    |-- client (機器人端的Pi程式碼備份)
    |-- camera_matrix.npy (鏡頭矯正矩陣檔)
    |-- follow_lane_api.py (主要執行)*
    |-- follow_lane_state.py (舊的主要執行-不需要了)
    |-- follow_lane.py (舊的主要執行-不需要了)
    |-- image_calibrator.py (鏡頭校正、透視轉換、挑選顏色範圍)
    |-- maze.py (ArUco tag、迷宮、判斷路口)
    |-- lanelines.py (尋找車道線)
    |-- README.md

    Raspberry Pi端 (車子)
    /home/pi/Python
    |-- find_pyaudio_device.py (列出Pi所連接的麥克風、喇叭裝置)
    |-- Demo
        |-- chatbot (聊天機器人)
            |-- chatbot.py  (聊天機器人主程式)
            |-- chatbot_use.py  (測試使用聊天機器人)
            |-- KnowledgeBase.json (語言庫)
            |-- log.txt (Log紀錄)
            |-- unknown.txt (未知問句紀錄)
        |-- epaper (電子紙顯示套件)
        |-- laneFollowing
            |-- laneFollowing.ino (車上Arduino檔案)
        |-- snowboy (喚醒詞套件)
        |-- cam_and_sound.py (鏡頭與聲音相關處理函式)
        |-- laneFollowing_state_api_client.py (主要執行)*
        |-- usb_pixel_ring_v2.py (麥克風陣列上的LED燈)

    Raspberry Pi端 (觀測地圖server)
    /home/pi/Python/camera_streaming
    |-- static (靜態檔案)
        |-- background.png (背景圖片)
        |-- jquery-3.4.1.min.js (jQuery套件)
    |-- templates (html模板)
        |-- index.html (html模板)
    |-- app.py (主要執行)*
    |-- base_camera.py (基本鏡頭函式)
    |-- camera_calibrator.py (影像處理)
    |-- camera_opencv.py (USB鏡頭函式)
    |-- camera_pi.py (Pi鏡頭函式)


### 遠端Raspberry pi端
電腦安裝VNC Viewer，皆須連接到同一個網域下。

Pi端 (車子) -> 192.168.0.147  
Pi端 (觀測地圖server) -> 192.168.0.185


### 安裝套件
**電腦端**

可以選擇安裝在系統中，或者安裝在使用 [pipenv][] 創的虛擬 Python 環境中。  
所需要之套件如下所示，或參考 [requirements.txt](./requirements.txt)。

    numpy==1.15.4+mkl
    opencv-contrib-python==4.0.0.21
    matplotlib==3.0.2
    Pillow==5.4.1


**Pi端 (車子)**

    gTTS==2.0.3
    gTTS-token==1.1.3
    numpy==1.15.4
    picamera==1.13
    PyAudio==0.2.11
    pygame==1.9.3
    SpeechRecognition==3.8.1
    Pillow==4.0.0 

** 在 Raspberry Pi 安裝 OpenCV 可安裝如Pi端(觀測地圖server)的opencv-contrib-python，  
或自行編譯，參考[gaborvecsei/install_opencv_raspberry_pi_3b.sh][]  
將此檔案放到Pi中，並用終端機執行：

    install_opencv_raspberry_pi_3b.sh

如果發生問題無法解決，可將第27行由`make -j2`改為`make`，改成用單顆處理器編譯。


**Pi端 (觀測地圖server)**

    Bootstrap-Flask==1.0.10
    Flask==0.12.1
    picamera==1.13
    Pillow==4.0.0
    numpy==1.15.4
    opencv-contrib-python==3.4.4.19


[pipenv]:  https://medium.com/@chihsuan/pipenv-%E6%9B%B4%E7%B0%A1%E5%96%AE-%E6%9B%B4%E5%BF%AB%E9%80%9F%E7%9A%84-python-%E5%A5%97%E4%BB%B6%E7%AE%A1%E7%90%86%E5%B7%A5%E5%85%B7-135a47e504f4 "Pipenv 更簡單、更快速的 Python 套件管理工具"
[gaborvecsei/install_opencv_raspberry_pi_3b.sh]:  https://gist.github.com/gaborvecsei/ad216f214731441bd66a34ae9a2dc3f3


### 初始一些設定
Pi車子跟電腦連接同一個WIFI，或者電腦開熱點分享給Pi車子連接。  
開啟電腦上的命令提示字元(CMD)輸入

    ipconfig

查看WIFI的IP為多少，去更改follow_lane_api.py(電腦)、laneFollowing_state_api_client.py(Pi車子)程式碼內的：

    TCP_IP = "你的IP"


如果Snowboy的Model有重新訓練下載的話，請把檔案移動到Pi車子專案內的

    snowboy/resources/models/檔名.pmdl

記得也要同步更改laneFollowing_state_api_client.py(Pi車子)程式碼：

    MODEL = "snowboy/resources/models/檔名.pmdl"


### 執行程式
**電腦端、Pi端 (車子)**
執行follow_lane_api.py(電腦)以及laneFollowing_state_api_client.py(Pi)程式碼。


**Pi端 (觀測地圖server) - 不是必要**

以終端機執行：

    python3 app.py runserver

如果沒錯誤，終端機會顯示連接IP，瀏覽器開啟IP:5000。  
** 如果網頁看不到東西，再重新執行



### Raspberry pi 系統IMG備份
因檔案很大，因此全部放雲端
https://drive.google.com/drive/folders/1soS5mUs9Lz7ItTppDXD1cxvOjFqyVfDO?usp=sharing

各版本簡介請看"IMG檔案版本內容.txt"

備份及還原軟體我是使用 [ImageWriter][]


[ImageWriter]:  https://sourceforge.net/projects/win32diskimager/


### 其他說明
* Gogs專案位置：http://140.125.32.134/Jia35/autoCar_lane_pi.git

* Pi端(車子) 裡的/home/pi/playRobot_car，是原先廠商附的範例程式碼。


* 如果Pi端(車子)麥克風無法收音，可到/home/pi/.asoundrc(隱藏檔案)查看內容是否為以下內容。

        pcm.!default {
            type hw
            playback.pcm {
                type hw
                slave.pcm "hw:0,0"
            }
            capture.pcm {
                type plug
                slave.pcm "hw:1,0"
            }
        }

        pcm.!default {
            type hw
            card 0
        }

        ctl.!default {
            type hw
            card 0
        }



### 參考網頁

#### Socket：
* [用 socket 將 OpenCV 影像傳送到遠端 client][]

#### 聊天機器人：
* [聊天機器人教學][]

#### 影像找車道：
* [GitHub - olala7846/CarND-Advanced-Lane-Lines][]
* [GitHub - asuri2/CarND-Advanced-Lane-Lines-P4][]
* [GitHub - olpotkin/CarND-Advanced-Lane-Lines][]
* [GitHub - hayoung-kim/Perception-for-Self-driving][]
* [Self-driving Cars — Advanced computer vision with OpenCV, finding lane lines][]

#### ArUco標籤系統：
* [ArUco markers][]

#### Snowboy 喚醒詞檢測引擎：
* [Snowboy Docs][]

#### DuckieTown：
* [DuckieTown官網][]
* [DuckieTown - Raspberry Pi台灣樹莓派][]

#### Donkey Car：
* [About Donkey Car][]



[用 socket 將 OpenCV 影像傳送到遠端 client]:  http://blog.maxkit.com.tw/2017/07/socket-opencv-client.html
[聊天機器人教學]:  https://www.codeproject.com/Articles/36106/Chatbot-Tutorial?fbclid=IwAR39GVi1U7vqaWUFwB9sxFDwmohiAlPJMkOI-5EMUBXpIlUT4RdtRQueAm4
[GitHub - olala7846/CarND-Advanced-Lane-Lines]:  https://github.com/olala7846/CarND-Advanced-Lane-Lines
[GitHub - asuri2/CarND-Advanced-Lane-Lines-P4]:  https://github.com/asuri2/CarND-Advanced-Lane-Lines-P4
[GitHub - olpotkin/CarND-Advanced-Lane-Lines]:  https://github.com/olpotkin/CarND-Advanced-Lane-Lines
[GitHub - hayoung-kim/Perception-for-Self-driving]:  https://github.com/hayoung-kim/Perception-for-Self-driving/tree/master/Lane-Line-Finding
[Self-driving Cars — Advanced computer vision with OpenCV, finding lane lines]:  https://chatbotslife.com/self-driving-cars-advanced-computer-vision-with-opencv-finding-lane-lines-488a411b2c3d
[ArUco markers]:  https://mecaruco2.readthedocs.io/en/latest/notebooks_rst/Aruco/aruco_basics.html?fbclid=IwAR13UClF8WLS7RQamX-jDmbaGWKVUc1uYJnjU6WvVU-nv6MBf70xTORy7jA
[Snowboy Docs]:  http://docs.kitt.ai/snowboy/
[DuckieTown官網]:  https://www.duckietown.org/
[DuckieTown - Raspberry Pi台灣樹莓派]:  https://www.raspberrypi.com.tw/tag/duckietown/
[About Donkey Car]:  http://docs.donkeycar.com/
"# YunBot" 
