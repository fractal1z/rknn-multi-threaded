import socket
import cv2
import numpy as np
import threading
import sys
# import yaml
# from ultralytics import YOLO
# from PyQt5.QtWidgets import *
import os
import pickle
import shutil
from queue import Queue
from func_det import draw
import concurrent.futures

def get_cam():
    '''这是从电脑摄像机获取视频的函数'''
    capture = cv2.VideoCapture(0)
    fps=capture.get(cv2.CAP_PROP_FPS)

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    hight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width,hight)
    while True:
        ret, frames = capture.read()  # ret为返回值，frame为视频的每一帧
        yield frames
        cv2.imshow('First view on the bullet', frames)
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break
    return frames

def read_yolo_txt_file(file_path):
    try:
        with open(file_path,'r') as file:
            lines=file.readlines()
    except FileNotFoundError:
        lines=[]
    annotations=[]
    for line in lines:
        line=line.strip().split(" ")
        if len(line)<5:
            break
        class_id=int(line[0])
        x_center,y_center,width,height=map(float,line[1:5])
        annotations.append((x_center,y_center,width,height,class_id))
    return annotations

def class_name(yaml_file):
    yolo_classes_file=open(yaml_file,encoding='utf-8')
    yolo_classes_string=yolo_classes_file.read()
    yolo_classes_dict=yaml.load(yolo_classes_string,Loader=yaml.FullLoader)
    yolo_classes=yolo_classes_dict['name']
    return yolo_classes


def yolo2voc(box,shape,resize):
    ##box: contains: (x_center,y_center,width,height,class_id)
    ##shape为原始图像的大小，resize为为了发送改变图像后的图像大小
    yolo_box=box
    image_width=shape[0]
    image_height=shape[1]
    center_x,center_y,width,height,class_id=yolo_box
    x_min=int((center_x-width/2)*image_width)*(resize[0]/image_width)
    y_min=int((center_y-height/2)*image_height)*(resize[1]/image_height)
    x_max=int((center_x+width/2)*image_width)*(resize[0]/image_width)
    y_max=int((center_y+height/2)*image_height)*(resize[1]/image_height)
    return (x_min,y_min,x_max,y_max,class_id)


def yoloxy2voc(box,cl,shape,resize,box_n):
    ##box: contains: (x_center,y_center,width,height,class_id)
    ##shape为原始图像的大小，resize为为了发送改变图像后的图像大小
    top, left, right, bottom = box
    image_width=shape[0]
    image_height=shape[1]
   # (int(left),int(top),int(right),int(bottom),int(cl))
   
    # x_min=int(top)
    # y_min=int(left)
    # x_max=int(right)
    # y_max=int(bottom)
    top=top-image_width/2
    right=right-image_width/2
    left=left-image_height/2
    bottom=bottom-image_height/2



    x_min=int(top+resize[0]/2)
    y_min=int(left+resize[1]/2)
    x_max=int(right+resize[0]/2)
    y_max=int(bottom+resize[1]/2)
    box_n.append((x_min,y_min,x_max,y_max))
    # print((x_min,y_min,x_max,y_max,cl))
    return (x_min,y_min,x_max,y_max,cl)


#
class Server():
    def __init__(self ,gs_ip,d_ip,port1,port2,port3,port4,port5):
        '''初始化函数'''
        super().__init__()
        # self.frames = frames
        # self.model=YOLO(model)
        self.port1=port1  ##发送数据的端口号
        self.port2=port2 ##接收数据的端口号
        self.port3=port3 ##发送目标识别结果的端口号
        self.port4=port4 ##发送对抗检测结果的端口号
        self.port5=port5 ##发送伪装检测结果的端口号
        self.ip1=gs_ip ##地面站的IP
        self.ip2=d_ip  ##自己的IP

        self.reg=None  ###判断是否接收到过目标识别的指令
        self.det=None  ###判断是否接收到过对抗检测的指令

        ##发送图像
        self.sock1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # ##发送文本
        self.sock3 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  ##发送目标识别结果
        self.sock4 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  ##发送对抗检测结果
        self.sock5 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  ##发送伪装检测结果

        ##接收
        self.sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock2.bind((self.ip2,self.port2))
            # 设置为非阻塞模式
        self.sock2.setblocking(False)
        self.frame_size=None
        self.max_trans_bit=640000
        self.img_quality=[int(cv2.IMWRITE_JPEG_QUALITY), 50]

        self.attack_statu=None #None,0,1,2

        cache_file=os.getcwd()+'/runs/detect/track'
        if os.path.exists(cache_file):
            shutil.rmtree(cache_file)
        else:
            pass

    def post_cam(self, frame):
        if self.frame_size==None:
            self.frame_size=(frame.shape[1],frame.shape[0])
        else:
            pass
        img_encode = cv2.imencode('.jpeg', frame,self.img_quality)[1]
        data_encode = np.array(img_encode)
        data = data_encode.tobytes()
        if sys.getsizeof(data)>self.max_trans_bit:
            pass
        else:
            self.sock1.sendto(data, (self.ip1,self.port1))

    # def run(self):
    #     receive_thread=threading.Thread(target=self.receive_command) ##开一个线程来监听指令
    #     receive_thread.start()

    #     for frame in self.frames:
    #         if self.reg==None and self.det==None:
    #             self.post_cam(frame)
    #         elif self.reg==True and self.det==None:
    #             self.post_cam_with_reg(frame)
    #         elif self.reg==None and self.det==True:
    #             self.post_cam_with_det(frame)
    #         elif self.reg==True and self.det==True:
    #             self.post_cam_with_both(frame)


    def post_cam_with_reg(self,frame):

        yolo_file=os.getcwd()+'/runs/detect/track/labels/image0.txt'
        if os.path.exists(yolo_file):
            os.remove(yolo_file)
        else:
            pass

        result=self.model.track(source=frame,conf=0.1,iou=0.5,persist=True,tracker='bytetrack.yaml',show=False,save_txt=True)
        object_info = read_yolo_txt_file(yolo_file)
        data1=[]
        ##通过VOC坐标格式来传递目标矩形框信息
        for object in object_info:
            data1.append(yolo2voc(object,frame.shape,self.frame_size))

        img_encode = cv2.imencode('.jpeg', frame,self.img_quality)[1]
        data_encode = np.array(img_encode)
        data = data_encode.tobytes()

        data1=pickle.dumps(data1)

        if sys.getsizeof(data)>self.max_trans_bit:
            pass
        else:
            self.sock3.sendto(data1,(self.ip1,self.port3))
            self.sock1.sendto(data, (self.ip1,self.port1))

    def post_cam_with_det(self,frame):
        ###放对抗检测算法的函数
        yolo_file=os.getcwd()+'/runs/detect/track/labels/image0.txt'
        if os.path.exists(yolo_file):
            os.remove(yolo_file)
        else:
            pass

        # result=self.model.track(source=frame,conf=0.1,iou=0.5,persist=True,tracker='bytetrack.yaml',show=False,save_txt=True)
        object_info = read_yolo_txt_file(yolo_file)
        data1=[]
        ##通过VOC坐标格式来传递目标矩形框信息
        for object in object_info:
            data1.append('存在对抗样本')

        img_encode = cv2.imencode('.jpeg', frame,self.img_quality)[1]
        data_encode = np.array(img_encode)
        data = data_encode.tobytes()

        data1=pickle.dumps(data1)

        if sys.getsizeof(data)>self.max_trans_bit:
            pass
        else:
            self.sock4.sendto(data1,(self.ip1,self.port4))
            self.sock1.sendto(data, (self.ip1,self.port1))

    def post_cam_with_both(self,frame):
        yolo_file = os.getcwd() + '/runs/detect/track/labels/image0.txt'
        if os.path.exists(yolo_file):
            os.remove(yolo_file)
        else:
            pass

        # result = self.model.track(source=frame, conf=0.1, iou=0.5, persist=True, tracker='bytetrack.yaml', show=False,
        #                           save_txt=True)
        object_info = read_yolo_txt_file(yolo_file)
        data1 = []
        data2=[]
        ##通过VOC坐标格式来传递目标矩形框信息
        for object in object_info:
            data1.append(yolo2voc(object, frame.shape, self.frame_size))
            data2.append('存在对抗样本')

        img_encode = cv2.imencode('.jpeg', frame, self.img_quality)[1]
        data_encode = np.array(img_encode)
        data = data_encode.tobytes()

        data1 = pickle.dumps(data1)
        data2= pickle.dumps(data2)

        if sys.getsizeof(data) > self.max_trans_bit:
            pass
        else:
            self.sock4.sendto(data2, (self.ip1, self.port4))
            self.sock3.sendto(data1, (self.ip1, self.port3))
            self.sock1.sendto(data, (self.ip1, self.port1))
    
    def send_data(self,sock, data, ip, port):
        sock.sendto(data, (ip, port))


    def post_result(self,frame,result_seg,result_tank,result_patch):
        
        frame_seg,boxes_seg,classes_seg,frame_seg_raw=result_seg
        frame_tank,boxes_tank,classes_tank,frame_tank_raw=result_tank
        frame_patch,boxes_patch,classes_patch,frame_patch_raw=result_patch
        data1 = []
        data2=[]
        data3=[]


        ##通过VOC坐标格式来传递目标矩形框信息tank
        if boxes_tank is not None:
            box_n=[]
            for box, cl in zip(boxes_tank, classes_tank):
                # print(box,cl,"\n")
                data1.append( yoloxy2voc(box,cl,frame_tank.shape,(640,480),box_n))
            # scores=np.ones(boxes_tank.size)
            # draw(frame_tank_raw, box_n, scores, classes_tank,("car","tank"))
            # cv2.imshow('frame_test_send', frame_tank_raw)
        # else:
        #     cv2.imshow('frame_test_send', frame_tank_raw)    

        ##通过VOC坐标格式来传递目标矩形框信息patch
        if boxes_patch is not None:
            box_n=[]
            for box, cl in zip(boxes_patch, classes_patch):
                # print(box,cl,"\n")
                data2.append( yoloxy2voc(box,cl,frame_patch.shape,(640,480),box_n))

        ##通过VOC坐标格式来传递目标矩形框信息伪装seg
        if boxes_seg is not None:
            box_n=[]
            for box, cl in zip(boxes_seg, classes_seg):
                # print(box,cl,"\n")
                data3.append( yoloxy2voc(box,cl,frame_seg.shape,(640,480),box_n))




        img_encode = cv2.imencode('.jpeg', frame_tank_raw, self.img_quality)[1]
        data_encode = np.array(img_encode)
        data = data_encode.tobytes()

        #print(data1)
        # data1 =  np.array(data1,dtype=np.int32)
        # print(data1)
        # data1=data1.tobytes()
        print(data1)
        data1= pickle.dumps(data1)
        data2= pickle.dumps(data2)
        data3= pickle.dumps(data3)

        if sys.getsizeof(data) > self.max_trans_bit:
            pass
        else:
            # self.sock3.sendto(data1, (self.ip1, self.port3))
            # self.sock4.sendto(data2, (self.ip1, self.port4))
            # self.sock1.sendto(data, (self.ip1, self.port1))
             # 创建并启动线程进行数据传输
            thread0 = threading.Thread(target=self.send_data, args=(self.sock5, data3, self.ip1, self.port5))
            thread1 = threading.Thread(target=self.send_data, args=(self.sock3, data1, self.ip1, self.port3))
            thread2 = threading.Thread(target=self.send_data, args=(self.sock4, data2, self.ip1, self.port4))
            thread3 = threading.Thread(target=self.send_data, args=(self.sock1, data, self.ip1, self.port1))
            

            # 启动线程
            thread0.start()
            thread1.start()
            thread2.start()
            thread3.start()

            # 等待所有线程完成
            thread0.join()
            thread1.join()
            thread2.join()
            thread3.join()
        

    def receive_command(self):
        while True:
            try:
                # 尝试接收数据
                data, addr = self.sock2.recvfrom(1024)
                # print(f"Received {data} from {addr}")
            except BlockingIOError:
                # 没有数据可用，处理其他任务
                return
            message=data.decode()
            print(message)
            if message=="目标识别":
                self.reg = True
            elif message=="对抗检测":
                self.det = True
            elif message=='关闭连接':
                self.reg=None
                self.det=None
            elif message=='攻击':
                self.attack_statu=0
            elif message=='取消攻击':
                self.attack_statu=1
            elif message=='再次确认':
                self.attack_statu=2



if __name__ == "__main__":
    frames = get_cam()
    S = Server(frames,'10.133.28.149','0.0.0.0',1234,8888,1200,1300,'best.pt')
    S.run()
# '10.133.28.149'
#192.168.20.85
