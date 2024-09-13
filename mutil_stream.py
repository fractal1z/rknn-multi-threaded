import cv2

def create_pipeline(pipeline_cmd):
    cap = cv2.VideoCapture(pipeline_cmd, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Failed to open pipeline!")
        return None
    return cap

# GStreamer 管道字符串
pipeline_cmd =  "udpsrc port=14552 buffer-size=819200 ! queue ! application/x-rtp,encoding-name=H264,payload=96 !  rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! capsfilter caps=\"video/x-raw,format=BGR\" ! videoconvert !  appsink sync=true"

# 创建多个视频接收管道
caps = [create_pipeline(pipeline_cmd) for _ in range(3)]  # 创建3个管道

while True:
    for cap in caps:
        if cap is None:
            continue
        ret, frame = cap.read()
        if ret:
            # 显示捕获的帧
            cv2.imshow("Received Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        continue
    break

for cap in caps:
    if cap:
        cap.release()
cv2.destroyAllWindows()
