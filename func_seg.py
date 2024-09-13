#以下代码改自https://github.com/rockchip-linux/rknn-toolkit2/tree/master/examples/onnx/yolov5
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from py_utils.coco_utils import COCO_test_helper,Letter_Box_Info
import torch

OBJ_THRESH, NMS_THRESH  = 0.5, 0.5
MAX_DETECT = 10
IMG_SIZE = (640, 640)
co_helper = COCO_test_helper(enable_letter_box=True)
# CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
#            "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
#            "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
#            "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
#            "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
#            "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
#            "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush ")

# CLASSES = ("pacth",)
# CLASSES = ("disguise_tank",)
# classes_choice = ("tank","car")
# 


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_process(position, anchors):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    col = col.repeat(len(anchors), axis=0)
    row = row.repeat(len(anchors), axis=0)
    anchors = np.array(anchors)
    anchors = anchors.reshape(*anchors.shape, 1, 1)

    box_xy = position[:,:2,:,:]*2 - 0.5
    box_wh = pow(position[:,2:4,:,:]*2, 2) * anchors

    box_xy += grid
    box_xy *= stride
    box = np.concatenate((box_xy, box_wh), axis=1)

    # Convert [c_x, c_y, w, h] to [x1, y1, x2, y2]
    xyxy = np.copy(box)
    xyxy[:, 0, :, :] = box[:, 0, :, :] - box[:, 2, :, :]/ 2  # top left x
    xyxy[:, 1, :, :] = box[:, 1, :, :] - box[:, 3, :, :]/ 2  # top left y
    xyxy[:, 2, :, :] = box[:, 0, :, :] + box[:, 2, :, :]/ 2  # bottom right x
    xyxy[:, 3, :, :] = box[:, 1, :, :] + box[:, 3, :, :]/ 2  # bottom right y

    return xyxy

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _crop_mask(masks, boxes):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """

    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)
    
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def seg_post_process(input_data, anchors):
    # input_data[0], input_data[2], and input_data[4] are detection box information
    # input_data[1], input_data[3], and input_data[5] are segmentation information
    # input_data[6] is the proto information
    boxes, scores, classes_conf = [], [], []
    # 1*255*h*w -> 3*85*h*w
    detect_part = [input_data[i*2].reshape([len(anchors[0]), -1]+list(input_data[i*2].shape[-2:])) for i in range(len(anchors))]
    seg_part = [input_data[i*2+1].reshape([len(anchors[0]), -1]+list(input_data[i*2+1].shape[-2:])) for i in range(len(anchors))]
    proto = input_data[-1]
    for i in range(len(detect_part)):
        boxes.append(box_process(detect_part[i][:, :4, :, :], anchors[i]))
        scores.append(detect_part[i][:, 4:5, :, :])
        classes_conf.append(detect_part[i][:, 5:, :, :])

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]
    seg_part = [sp_flatten(_v) for _v in seg_part]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)
    seg_part = np.concatenate(seg_part)

    # filter according to threshold
    boxes, classes, scores, seg_part = filter_boxes(boxes, scores, classes_conf, seg_part)

    # zipped = zip(boxes, classes, scores, seg_part)
    # sort_zipped = sorted(zipped, key=lambda x: (x[2]), reverse=True)
    # result = zip(*sort_zipped)

    # max_nms = 30000
    # n = boxes.shape[0]  # number of boxes
    # if not n:
    #     return None, None, None, None
    # elif n > max_nms:  # excess boxes
    #     boxes, classes, scores, seg_part = [np.array(x[:max_nms]) for x in result]
    # else:
    #     boxes, classes, scores, seg_part = [np.array(x) for x in result]

    # nms
    nboxes, nclasses, nscores, nseg_part = [], [], [], []
    # agnostic = 0
    # max_wh = 7680
    # c = classes * (0 if agnostic else max_wh)
    # ids = torchvision.ops.nms(torch.tensor(boxes, dtype=torch.float32) + torch.tensor(c, dtype=torch.float32).unsqueeze(-1),
    #                           torch.tensor(scores, dtype=torch.float32), NMS_THRESH)
    # real_keeps = ids.tolist()[:MAX_DETECT]
    # nboxes.append(boxes[real_keeps])
    # nclasses.append(classes[real_keeps])
    # nscores.append(scores[real_keeps])
    # nseg_part.append(seg_part[real_keeps])
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        se  = seg_part[inds]
        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])
        nseg_part.append(se[keep])

    if not nclasses and not nscores:
        return None, None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    seg_part = np.concatenate(nseg_part)

    ph, pw = proto.shape[-2:]
    proto = proto.reshape(seg_part.shape[-1], -1)

    seg_img=1##nosegdraw
    # seg_img = np.matmul(seg_part, proto)
    # seg_img = sigmoid(seg_img)
    # seg_img = seg_img.reshape(-1, ph, pw)

    # seg_threadhold = 0.5

    # # crop seg outside box
    # seg_img = F.interpolate(torch.tensor(seg_img)[None], torch.Size([640, 640]), mode='bilinear', align_corners=False)[0]
    # seg_img_t = _crop_mask(seg_img,torch.tensor(boxes) )

    # seg_img = seg_img_t.numpy()
    # seg_img = seg_img > seg_threadhold

    return boxes, classes, scores, seg_img
# def process(input, mask, anchors):

#     anchors = [anchors[i] for i in mask]
#     grid_h, grid_w = map(int, input.shape[0:2])

#     box_confidence = input[..., 4]
#     box_confidence = np.expand_dims(box_confidence, axis=-1)

#     box_class_probs = input[..., 5:]

#     box_xy = input[..., :2] *2 - 0.5

#     col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
#     row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
#     col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
#     row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
#     grid = np.concatenate((col, row), axis=-1)
#     box_xy += grid
#     box_xy *= int(IMG_SIZE/grid_h)

#     box_wh = pow(input[..., 2:4] *2, 2)
#     box_wh = box_wh * anchors

#     return np.concatenate((box_xy, box_wh), axis=-1), box_confidence, box_class_probs


def filter_boxes(boxes, box_confidences, box_class_probs, seg_part):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    seg_part = (seg_part * box_confidences.reshape(-1, 1))[_class_pos]

    return boxes, classes, scores, seg_part

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    return np.array(keep)


# def yolov5_post_process(input_data):
#     masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
#     anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
#                [59, 119], [116, 90], [156, 198], [373, 326]]
#     # anchors = [[3.4550781,2.1660156], [4.453125,3.0683594], [5.4726562,4.0195312], [6.8007812,5.9375], [9.3046875,6.5507812],
#     #            [9.4921875,8.5390625], [11.8671875,9.1328125], [13.1171875,12.921875], [23.078125,21.296875]]#path

#     boxes, classes, scores = [], [], []
#     for input, mask in zip(input_data, masks):
#         b, c, s = process(input, mask, anchors)
#         b, c, s = filter_boxes(b, c, s)
#         boxes.append(b)
#         classes.append(c)
#         scores.append(s)

#     boxes = np.concatenate(boxes)
#     boxes = xywh2xyxy(boxes)
#     classes = np.concatenate(classes)
#     scores = np.concatenate(scores)

#     nboxes, nclasses, nscores = [], [], []
#     for c in set(classes):
#         inds = np.where(classes == c)
#         b = boxes[inds]
#         c = classes[inds]
#         s = scores[inds]

#         keep = nms_boxes(b, s)

#         nboxes.append(b[keep])
#         nclasses.append(c[keep])
#         nscores.append(s[keep])

#     if not nclasses and not nscores:
#         return None, None, None

#     return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)


def draw(image, boxes, scores, classes,classes_choice):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)
        if cl<0:
            cv2.rectangle(image, (top, left), (int(right), int(bottom)), (255, 0, 0), 2)
        else:
            cv2.rectangle(image, (top, left), (int(right), int(bottom)), (0, 0, 255), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(classes_choice[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 2)

def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)  # add border
    co_helper.letter_box_info_list.append(Letter_Box_Info(shape,new_shape, r, r, dw, dh, color))
    return im
    # return im, ratio, (dw, dh)

def merge_seg(image, seg_img, classes):
    color = Colors()
    for i in range(len(seg_img)):
        seg = seg_img[i]
        seg = seg.astype(np.uint8)
        seg = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)
        seg = seg * color(classes[i])
        seg = seg.astype(np.uint8)
        image = cv2.add(image, seg)
    return image

def myFunc_seg(rknn_lite, IMG,anchors,classes):

    classes_choice=classes
    anchors = [[[10, 13], [16, 30], [33, 23]], 
               [[30, 61], [62, 45],[59, 119]],
                [[116, 90], [156, 198], [373, 326]]]


#   anchors = [[3.4550781,2.1660156], [4.453125,3.0683594], [5.4726562,4.0195312], [6.8007812,5.9375], [9.3046875,6.5507812],
#              [9.4921875,8.5390625], [11.8671875,9.1328125], [13.1171875,12.921875], [23.078125,21.296875]]#path
    
    # 等比例缩放
    IMG_raw=IMG.copy()
    IMG = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
    IMG2 = letterbox(IMG)
    
    # 强制放缩
    # IMG = cv2.resize(IMG, (IMG_SIZE, IMG_SIZE))
    IMG_2 = np.expand_dims(IMG2, 0)
    outputs = rknn_lite.inference(inputs=[IMG_2])


    # input0_data = outputs[0].reshape([3, -1]+list(outputs[0].shape[-2:]))
    # input1_data = outputs[1].reshape([3, -1]+list(outputs[1].shape[-2:]))
    # input2_data = outputs[2].reshape([3, -1]+list(outputs[2].shape[-2:]))

    # input_data = list()
    # input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
    # input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
    # input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

    # boxes, classes, scores = yolov5_post_process(input_data)
    boxes, classes, scores, seg_img = seg_post_process(outputs, anchors)

    IMG = cv2.cvtColor(IMG, cv2.COLOR_RGB2BGR)
    
    if boxes is not None:
        real_boxs = co_helper.get_real_box(boxes)
        # real_segs = co_helper.get_real_seg(seg_img)
        # IMG = merge_seg(IMG, real_segs, classes)
        draw(IMG, real_boxs, scores, classes,classes_choice)
    return IMG, boxes, classes,IMG_raw
