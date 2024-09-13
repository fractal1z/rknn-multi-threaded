import cv2

# 使用 GStreamer 管道创建视频捕获对象
gst_pipeline = (
    'tcpclientsrc host=192.168.185.163 port=5000 ! '
    'queue ! '
    'h264parse ! '
    'avdec_h264 ! '
    'videoconvert ! '
    'appsink'
)
pipeline_cmd =  "udpsrc multicast-group=224.1.1.1 auto-multicast=true port=5000  buffer-size=409600 ! queue ! application/x-rtp,encoding-name=H264,payload=96 !  rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! capsfilter caps=\"video/x-raw,format=BGR\" ! videoconvert !  appsink sync=true"

cap = cv2.VideoCapture(pipeline_cmd, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Failed to open video stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    cv2.imshow('Video Stream2', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
