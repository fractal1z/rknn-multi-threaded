import cv2
import time
from rknnpool import rknnPoolExecutor
# 图像处理函数，实际应用过程中需要自行修改
from func_seg import myFunc_seg

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
pipeline_cmd =  "udpsrc port=14552 buffer-size=819200 ! queue ! application/x-rtp,encoding-name=H264,payload=96 !  rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! capsfilter caps=\"video/x-raw,format=BGR\" ! videoconvert !  appsink sync=true"
cap = cv2.VideoCapture(pipeline_cmd, cv2.CAP_GSTREAMER)
# cap = cv2.VideoCapture('./sigma.mp4')
# cap = cv2.VideoCapture(0)
# modelPath = "./rknnModel/yolov5s_relu_tk2_RK3588_i8.rknn"
modelPath = "./rknnModel/seg_8_26.rknn"
# 线程数, 增大可提高帧率
TPEs = 3
# 初始化rknn池
pool = rknnPoolExecutor(
    rknnModel=modelPath,
    TPEs=TPEs,
    func=myFunc_seg)

# 初始化异步所需要的帧
if (cap.isOpened()):
    for i in range(TPEs + 1):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            del pool
            exit(-1)
        pool.put(frame)

# 获取视频的基本信息
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fpsv = 30
print(frame_height,frame_width)
# 定义输出视频文件的编码方式和文件名
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 你可以使用其他编码器，如 'XVID', 'MJPG'
out_filename = 'output.mp4'

# 初始化 VideoWriter 对象
writer = cv2.VideoWriter(out_filename, fourcc, fpsv, (640,480))


frames, loopTime, initTime ,fps= 0, time.time(), time.time(),0
while (cap.isOpened()):
    frames += 1
    ret, frame = cap.read()
    if not ret:
        break

    # pool.put(frame)
    # frame, flag = pool.get()
    # if flag == False:
    #     break

    height, width, channels = frame.shape
    # print(f"Frame size: {width}x{height}, Channels: {channels}")
    text = f"FPS: {fps:.2f}"
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness)
    cv2.imshow('test', frame)
    writer.write(frame)
    
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
pool.release()
