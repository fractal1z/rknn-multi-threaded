import cv2
import time
import socket
import pickle
from rknnpool import rknnPoolExecutor
# 图像处理函数，实际应用过程中需要自行修改
from func_seg import myFunc_seg
from func_det import myFunc_det
from send_video import Server
import threading

fontScale = 0.6
color=(0, 0, 0)
thickness = 2
org=(10, 110)

# 使用 GStreamer 管道创建视频捕获对象
gst_pipeline = (
    'tcpclientsrc host=127.0.0.1 port=5000 ! '
    'queue ! '
    'h264parse ! '
    'avdec_h264 ! '
    'videoconvert ! '
    'appsink'
)

pipeline_cmd =  "udpsrc port=14552 buffer-size=409600 ! queue ! \
    application/x-rtp,encoding-name=H264,payload=96 !  rtph264depay ! h264parse ! \
    avdec_h264 ! videoconvert ! capsfilter caps=\"video/x-raw,format=BGR\" ! videoconvert !  appsink sync=true"


cap = cv2.VideoCapture(pipeline_cmd, cv2.CAP_GSTREAMER)
# cap = cv2.VideoCapture('./test_8_28.mp4')
# cap = cv2.VideoCapture(0)
modelPath_patch = "/home/orangepi/rknn-multi-threaded/rknnModel/best_patch8.29.rknn"
modelPath_tank = "/home/orangepi/rknn-multi-threaded/rknnModel/tank_car_9_8.rknn"
modelPath_seg = "/home/orangepi/rknn-multi-threaded/rknnModel/seg_9_8.rknn"

# 线程数, 增大可提高帧率
TPEs = 3

# 初始化rknn池_seg
classes_choice_seg= ("disguise_tank",)
anchors_seg = [[[10, 13], [16, 30], [33, 23]], 
               [[30, 61], [62, 45],[59, 119]],
                [[116, 90], [156, 198], [373, 326]]]
pool_seg = rknnPoolExecutor(
    rknnModel=modelPath_seg,
    TPEs=TPEs,
    func=myFunc_seg)
# 初始化异步所需要的帧
if (cap.isOpened()):
    for i in range(TPEs + 1):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            del pool_seg
            exit(-1)
        pool_seg.put(frame,anchors_seg,classes_choice_seg)



# 初始化rknn池_tank
classes_choice_tank = ("car","tank")
anchors_tank = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
    [59, 119], [116, 90], [156, 198], [373, 326]]
pool_tank = rknnPoolExecutor(
    rknnModel=modelPath_tank,
    TPEs=TPEs,
    func=myFunc_det)
# 初始化异步所需要的帧
if (cap.isOpened()):
    for i in range(TPEs + 1):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            del pool_tank
            exit(-1)
        pool_tank.put(frame,anchors_tank,classes_choice_tank)


# 初始化rknn池_patch
classes_choice_patch =  ("pacth", )
anchors_patch = [[2.859375,1.7851562], [4.1445312,2.265625], [4.6679688,3.1796875], [6.2695312,3.171875], [5.6289062,4.6015625],
            [7.8945312,4.46875], [6.78125,6.5039062], [9.4140625,6.4414062], [11.09375,8.546875]]#path
pool_patch = rknnPoolExecutor(
    rknnModel=modelPath_patch,
    TPEs=TPEs,
    func=myFunc_det)
# 初始化异步所需要的帧
if (cap.isOpened()):
    for i in range(TPEs + 1):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            del pool_patch
            exit(-1)
        pool_patch.put(frame,anchors_patch,classes_choice_patch)


# 获取视频的基本信息
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fpsv = 30
print(frame_height,frame_width)
# 定义输出视频文件的编码方式和文件名
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 你可以使用其他编码器，如 'XVID', 'MJPG'
out_filename = 'output.mp4'

# 初始化 VideoWriter 对象
writer = cv2.VideoWriter(out_filename, fourcc, 20, (640,640))


frames, loopTime, initTime ,fps= 0, time.time(), time.time(),0
S=Server('192.168.1.124','192.168.1.128',1234,8888,1200,1300,1400)
while (cap.isOpened()):
    frames += 1
    ret, frame = cap.read()
    if not ret:
        break
    # height, width, channels = frame.shape
    # print(f"Frame size: {width}x{height}, Channels: {channels}")

    # frame = cv2.imread('/home/orangepi/rknn-multi-threaded/test.jpg')

    if frames % 3 == 0:## 隔几帧发一下消息
        pool_seg.put(frame.copy(),anchors_seg,classes_choice_seg)
        result_seg, flag = pool_seg.get()
        frame_seg,boxes_seg,classes_seg,_=result_seg
        # print(boxes_seg,classes_seg)
        if flag == False:
            break
        cv2.imshow('frame_seg', frame_seg)

        pool_tank.put(frame.copy(),anchors_tank,classes_choice_tank)
        result_tank, flag = pool_tank.get()
        frame_tank,boxes_tank,classes_tank,_=result_tank
        if flag == False:
            break
        cv2.imshow('frame_tank', frame_tank)

        frame = cv2.resize(frame, (640, 480))
        pool_patch.put(frame.copy(),anchors_patch,classes_choice_patch)
        result_patch, flag = pool_patch.get()
        frame_patch,boxes_patch,classes_patch,_=result_patch
        if flag == False:
            break
        cv2.imshow('frame_patch', frame_patch)

    #S.post_cam(frame)
    
        S.post_result(frame.copy(),result_seg,result_tank,result_patch)

    receive_thread=threading.Thread(target=S.receive_command) ##开一个线程来监听指令
    receive_thread.start()
    # S.receive_command()
    # print(S.reg)

   
    text = f"FPS: {fps:.2f}"
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness)
    #cv2.imshow('test', frame)
    #writer.write(frame_tank)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # print(frames)
    if frames % 30 == 0:
        fps=30 / (time.time() - loopTime)
        print("\n\n---------------------30帧内平均帧率:" ,fps, "fps/s---------------------",  "\n\n")
        loopTime = time.time()

print("总平均帧率\t", frames / (time.time() - initTime))
# 释放cap和rknn线程池
cap.release()
cv2.destroyAllWindows()

