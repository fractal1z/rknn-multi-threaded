import cv2

# 使用 GStreamer 管道创建视频捕获对象
pipeline_cmd = "udpsrc port=14552 buffer-size=40960000 ! queue ! application/x-rtp,encoding-name=H264,payload=96 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! capsfilter caps=\"video/x-raw,format=BGR\" ! videoconvert ! appsink sync=true"

cap = cv2.VideoCapture(pipeline_cmd, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Failed to open video stream")
    exit()

# 创建 GStreamer 管道来保存视频流到本地文件
local_pipeline_cmd = "appsrc ! videoconvert ! x264enc bitrate=5000 ! mp4mux ! filesink location=output1.mp4"

# 使用 GStreamer 管道创建视频推送对象
push_pipeline = cv2.VideoWriter(local_pipeline_cmd, cv2.CAP_GSTREAMER, 0, 30, (640, 480), True)

if not push_pipeline.isOpened():
    print("Failed to open local file stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # 将帧写入本地文件
    push_pipeline.write(frame)
    
    cv2.imshow('Video Stream', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
push_pipeline.release()
cv2.destroyAllWindows()
