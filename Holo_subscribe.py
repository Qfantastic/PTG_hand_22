# from tomlkit import key
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim
from torchvision import datasets, models, transforms
import copy

import message_filters
import cv2
import threading
import argparse
import numpy as np
import rospy

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import PIL
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

lock = threading.Lock()


# import argparse

import sys
from pathlib import Path

# import colorsys


import os

# os.environ['PYOPENGL_PLATFORM'] = 'egl'# 'osmesa'

from torchvision.utils import make_grid

from utils.augmentations import letterbox
from utils.torch_utils import select_device, time_sync
from models.common import DetectMultiBackend
from utils.general import (
    LOGGER,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_coords,
    strip_optimizer,
    xyxy2xywh,
)
from utils.plots import Annotator, colors, save_one_box

curr_dir = os.path.dirname(os.path.abspath(__file__))

class_names = np.array(
    [
        "bowl",
        "cake_bowl",
        "coffee_bean",
        "cup",
        "cutting _board",
        "filter_cone",
        "filter",
        "floss",
        "grinder",
        "grinder_lip",
        "jelly",
        "kettle",
        "kettle_lip",
        "knife",
        "measure_cup",
        "mug",
        "nut_better",
        "paper_towel",
        "plate",
        "scissors",
        "spoon",
        "thermometer",
        "toothpick",
        "tortilla",
        "kitchen_scale",
        "whisk",
        "zip_top_bag",
    ]
)

# object_list = [
#     "kettle",
#     "measuring_cup",
#     "mug",
#     "kettle_lid",
#     "filter_cone",
#     "paper_filter_circle",
#     "paper_filter_half",
#     "paper_filter_quarter",
#     "kitchen_scale",
#     "coffee_beans",
#     "coffee_grinder",
#     "coffee_grinder_lid",
#     "coffee_grounds",
#     "thermometer",
# ]

object_list = ['cutting_board', 'open_tortilla', 'rolled_tortilla', 'sliced_tortilla', 'knife', 
'peanut_butter_jar', 'paper_towel', 'jelly_jar', 'toothpicks', 'floss_container', 'plate', 'kettle', 'measuring_cup', 'mug', 
'kettle_lid', 'filter_cone', 'paper_filter_circle', 'paper_filter_half', 'paper_filter_quarter', 'kitchen_scale', 'coffee_beans', 
'coffee_grinder', 'coffee_grinder_lid', 'coffee_grounds', 'thermometer', 'paper_cake_liner', 'bowl', 'flour', 'sugar', 'baking_powder', 'salt', 'whisk', 'oil', 'water', 'vanilla', 
'batter', 'microwave', 'cake', 'measuring_spoon', 'zip_top_bag', 'chocolate_frosting', 'scissors']


depth_h = 512
depth_w = 512
tf_avg = np.load(os.path.join(curr_dir, "./data/tf_avg.npy"))
lut_file = os.path.join(curr_dir, "./data/calibration/depth_lut.bin")
with open(lut_file, mode="rb") as lut_file:
    lut = np.frombuffer(lut_file.read(), dtype="f")
    lut = np.reshape(lut, (-1, 3))


def world2pixel_box(x, intrinsic):
    fx = intrinsic[0]
    fy = intrinsic[1]
    ux = intrinsic[2]
    uy = intrinsic[3]

    x[:, 0] = x[:, 0] * fx / x[:, 2] + ux
    x[:, 1] = x[:, 1] * fy / x[:, 2] + uy
    return x[:, 0:2]


def world2pixel(x, intrinsic):
    t = x.copy()
    fx = intrinsic[0]
    fy = intrinsic[1]
    ux = intrinsic[2]
    uy = intrinsic[3]

    t[0] = x[0] * fx / x[2] + ux
    t[1] = x[1] * fy / x[2] + uy
    return t[0:2]


def bounding_box_to_pixel(x, intrinsic, range_box):
    sign = [1, -1]
    bounding_box_w = np.zeros((8, 3))
    bounding_box_p = np.zeros((4, 2))
    n = 0
    for i in range(2):
        for j in range(2):
            for m in range(2):
                bounding_box_w[n, 0] = x[0] + sign[i] * range_box
                bounding_box_w[n, 1] = x[1] + sign[j] * range_box
                bounding_box_w[n, 2] = x[2] + sign[m] * 0.6 * range_box
                n = n + 1

    result = world2pixel_box(bounding_box_w, intrinsic)

    l_x = np.zeros(8)
    l_y = np.zeros(8)
    for j in range(8):
        l_x[j] = result[j, 0]
        l_y[j] = result[j, 1]
    bound_x_min = np.min(l_x)
    bound_x_max = np.max(l_x)
    bound_y_min = np.min(l_y)
    bound_y_max = np.max(l_y)

    bounding_box_p[0, :] = np.array([bound_x_min, bound_y_min])
    bounding_box_p[1, :] = np.array([bound_x_max, bound_y_min])
    bounding_box_p[2, :] = np.array([bound_x_max, bound_y_max])
    bounding_box_p[3, :] = np.array([bound_x_min, bound_y_max])
    return bounding_box_p


def holo_2D_to_3D(
    abc,
    soft_max,
    lut,
    center,
    depth_image_or,
    intrinsic,
    range_box,
    tf,
    abc_or,
    model_cls,
    device,
    pre_process,
):
    height_color = int(abc.shape[0])
    width_color = int(abc.shape[1])

    depth_image = depth_image_or / 1000
    depth_image = np.tile(depth_image.flatten().reshape((-1, 1)), (1, 3))
    points = depth_image * lut
    n_center = center[1] * depth_w + center[0]
    hand_center_3D_depth = np.append(points[n_center], 1)
    hand_center_3D_color = np.dot(tf, hand_center_3D_depth)[0:3]

    bounding_box_p = bounding_box_to_pixel(hand_center_3D_color, intrinsic, range_box)
    bounding_box_p_d = bounding_box_to_pixel(hand_center_3D_color, intrinsic, 0.10)
    center_p = world2pixel(hand_center_3D_color, intrinsic)

    image_to_draw = abc.copy()

    c = 0.75

    x1 = int(bounding_box_p[0][0])
    y1 = int(bounding_box_p[0][1])
    x2 = int(bounding_box_p[2][0])
    y2 = int(bounding_box_p[2][1])

    w = x2 - x1
    h = y2 - y1

    x1_d = int(bounding_box_p_d[0][0])
    y1_d = int(bounding_box_p_d[0][1])
    x2_d = int(bounding_box_p_d[2][0])
    y2_d = int(bounding_box_p_d[2][1])

    w_d = x2_d - x1_d
    h_d = y2_d - y1_d

    if (
        (width_color - x1) > c * w
        and (height_color - y1) > c * h
        and x2 > c * w
        and y2 > c * h
    ):
        crop_color = abc_or[
            max(y1_d, 0) : min(y2_d, height_color),
            max(x1_d, 0) : min(x2_d, width_color),
        ]

        crop_color = PIL.Image.fromarray(crop_color)
        crop_color = pre_process(crop_color)
        crop_color = crop_color.unsqueeze(0)
        crop_color = crop_color.to(device)
        outputs = model_cls(crop_color)
        _, pre_crop = torch.max(outputs.data, 1)
        class_index = int(pre_crop.cpu().numpy())
        print(class_names[class_index])
        cv2.rectangle(
            image_to_draw,
            (max(x1, 0), max(y1, 0)),
            (min(x2, width_color), min(y2, height_color)),
            (0, 0, 255),
            3,
        )
        if torch.max(soft_max(outputs.data)).cpu().numpy() > 0.7:
            cv2.putText(
                image_to_draw,
                class_names[class_index],
                (max(x1, 0), max(y1, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                3,
                cv2.LINE_AA,
            )

    print("center_p", center_p)
    return image_to_draw


class ImageListener:
    def __init__(self, topics, node_id="image_listener_hod", slop_seconds=0.2):

        self.cv_bridge = CvBridge()
        self.node_id = node_id
        self.topics = topics

        # self.queue_size = 2 * len(topics)
        self.queue_size = 3
        self.slop_seconds = slop_seconds
        self.color = None
        # self.depth_or = None
        self.depth = None
        self.depth_frame_id = None
        self.depth_frame_stamp = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None
        self.K_mat = None
        self.empty_label = np.zeros((176, 176, 3), dtype=np.uint8)

        self.synced_msgs = None

        # initialize a node
        rospy.init_node(self.node_id, anonymous=True)

        self.box_pub = rospy.Publisher("box_label", Image, queue_size=10)
        self.color_box_pub = rospy.Publisher("color_box_label", Image, queue_size=10)

        # self.holo_depth = message_filters.Subscriber(topics[0], Image, queue_size=10)
        # self.holo_color = message_filters.Subscriber(topics[1], Image, queue_size=10)

        self.holo_subs = [
            message_filters.Subscriber(t, Image, queue_size=10) for t in topics[:-1]
        ]

        self.holo_subs.append(
            message_filters.Subscriber(topics[-1], CameraInfo, queue_size=10)
        )

        ts = message_filters.ApproximateTimeSynchronizer(
            self.holo_subs, self.queue_size, self.slop_seconds
        )
        ts.registerCallback(self.ts_callback)

    def ts_callback(self, *msg):

        # self.synced_msgs = msg

        depth_msg, color_msg, caminfo_msg = msg
        depth_cv = self.cv_bridge.imgmsg_to_cv2(depth_msg, depth_msg.encoding).astype(
            "uint16"
        )
        color_cv = self.cv_bridge.imgmsg_to_cv2(color_msg, color_msg.encoding)
        # get the intrinsic matrix of rgb camera
        K_mat = np.array(caminfo_msg.P, dtype=np.float32).reshape((3, 4))[:3, :3]

        with lock:
            self.color = color_cv.copy()
            self.depth = depth_cv.copy()
            self.rgb_frame_id = color_msg.header.frame_id
            self.rgb_frame_stamp = color_msg.header.stamp
            self.depth_frame_id = depth_msg.header.frame_id
            self.depth_frame_stamp = depth_msg.header.stamp
            self.K_mat = K_mat

        #     self.depth_or = depth_cv.copy()
        #     #self.depth = depth_cv.copy()
        #     self.depth_frame_id = depth.header.frame_id
        #     self.depth_frame_stamp = depth.header.stamp

    def run_network(
        self,
        model,
        m,
        index,
        model_cls,
        model_obj,
        device,
        pre_process,
        imgsz=(640, 640),
        augment=False,
        visualize=False,
        conf_thres=0.4,
        iou_thres=0.4,
        classes=None,
        agnostic_nms=False,
        max_det=1000,
        line_thickness=3,
        hide_labels=False,
        hide_conf=False,
    ):

        with lock:
            # if listener.im is None:
            #     return
            if self.depth is None:
                return

            depth_or_img = self.depth.copy()
            depth_frame_id = self.depth_frame_id
            depth_frame_stamp = self.depth_frame_stamp

            color_or_img = self.color.copy()
            color_frame_id = self.rgb_frame_id
            color_frame_stamp = self.depth_frame_stamp

            K_mat = self.K_mat

        #     depth_or_img = np.array(self.depth_or).copy()
        #     #depth_img = np.array(self.depth_or).copy()
        #     depth_frame_id = self.depth_frame_id
        #     depth_frame_stamp = self.depth_frame_stamp

        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)

        # if self.synced_msgs is not None:
        if depth_or_img.ndim == 2:
            # depth_msg, color_msg, caminfo_msg = self.synced_msgs
            # depth_or_img = self.cv_bridge.imgmsg_to_cv2(depth_msg,depth_msg.encoding).astype('uint16')
            # depth_frame_id = depth_msg.header.frame_id
            # depth_frame_stamp = depth_msg.header.stamp
            #
            # color_or_img = self.cv_bridge.imgmsg_to_cv2(color_msg,color_msg.encoding)
            # # color_frame_id = color_msg.header.frame_id
            # # color_frame_stamp = color_msg.header.stamp
            #
            # # get the intrinsic matrix of rgb camera
            # K_mat = np.array(caminfo_msg.P, dtype=np.float32).reshape((3,4))[:3,:3]
            # # print(K_mat)

            # Do the normalization
            # print('max:', np.max(depth_or_img), 'min:', np.min(depth_or_img))
            range_img = np.max(depth_or_img) - np.min(depth_or_img)
            depth_img = ((depth_or_img / range_img) * 255).astype("uint8")
            stacked_img = np.stack((depth_img,) * 3, axis=-1)
            # print('max:', np.max(stacked_img), 'min:', np.min(stacked_img))
            img = letterbox(stacked_img, imgsz, stride=stride, auto=pt)[0]
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            # print('max:', np.max(img), 'min:', np.min(img))
            img = np.ascontiguousarray(img)

            bs = 1
            model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))

            # print('max:', np.max(img), 'min:', np.min(img))
            img = torch.from_numpy(img).to(device)
            img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
            img /= 255  # 0 - 255 to 0.0 - 1.0

            if len(img.shape) == 3:
                img = img[None]
            # print(img.shape)

            # predict the hand on depth images
            pred = model(img, augment=augment, visualize=visualize)
            pred = non_max_suppression(
                pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
            )

            print("prediction:")

            # print(pred)

            outputs = predictor(color_or_img)
            output_fields = outputs["instances"].to("cpu")._fields

            pred_boxes = output_fields["pred_boxes"].tensor.numpy()
            pred_boxes_shape = pred_boxes.shape
            pred_classes = (
                output_fields["pred_classes"].numpy().reshape(pred_boxes_shape[0], 1)
            )
            scores = output_fields["scores"].numpy().reshape(pred_boxes_shape[0], 1)

            answer = copy.deepcopy(pred_boxes)
            answer = np.concatenate((answer, pred_classes), axis=1)
            answer = np.concatenate((answer, scores), axis=1)

            color_or_img_rgb = cv2.cvtColor(color_or_img.copy(), cv2.COLOR_BGR2RGB)
            image_draw = cv2.cvtColor(color_or_img.copy(), cv2.COLOR_BGR2RGB)

            for i in range(answer.shape[0]):
                # cv2.rectangle(image_draw, (int(answer[i][0]), int(answer[i][1])),
                #               (int(answer[i][2]), int(answer[i][3])), (0, 0, 255), 3)
                center_obj = np.array(
                    [
                        int((answer[i][0] + answer[i][2]) / 2),
                        int((answer[i][1] + answer[i][3]) / 2),
                    ]
                )
                textsize = cv2.getTextSize(
                    object_list[int(answer[i][4])], cv2.FONT_HERSHEY_SIMPLEX, 1, 2
                )[0]
                cv2.putText(
                    image_draw,
                    object_list[int(answer[i][4])],
                    (
                        int(center_obj[0] - 0.5 * textsize[0]),
                        int(center_obj[1] - 0.5 * textsize[1]),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            for i, det in enumerate(pred):
                im0 = stacked_img.copy()
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                annotator = Annotator(
                    im0, line_width=line_thickness, example=str(names)
                )
                # color_abc = cv2.cvtColor(color_or_img.copy(), cv2.COLOR_BGR2RGB)
                # color_or_img_rgb = cv2.cvtColor(color_or_img.copy(), cv2.COLOR_BGR2RGB)

                if len(det):
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape
                    ).round()

                    for *xyxy, conf, cls in reversed(det):

                        c = int(cls)
                        xywh = (
                            (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                            .view(-1)
                            .tolist()
                        )
                        label = (
                            None
                            if hide_labels
                            else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        )

                        center_x = int(xywh[0] * depth_w)
                        center_y = int(xywh[1] * depth_h)
                        center = np.array([center_x, center_y])
                        print("xyxy:", xyxy)
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        holo_color_intrinsic = np.array(
                            [K_mat[0][0], K_mat[1][1], K_mat[0][2], K_mat[1][2]]
                        )

                        # color_bounding_box
                        # color_abc = holo_2D_to_3D(color_abc,color_or_img, lut, center, depth_or_img, holo_color_intrinsic, 0.08,
                        #                           tf_avg,color_or_img_rgb,model_cls, device, pre_process)

                        image_draw = holo_2D_to_3D(
                            image_draw,
                            m,
                            lut,
                            center,
                            depth_or_img,
                            holo_color_intrinsic,
                            0.08,
                            tf_avg,
                            color_or_img_rgb,
                            model_cls,
                            device,
                            pre_process,
                        )

                        # color_xyxy , color_label, color_cls = holo_2D_to_3D(color_abc, lut, center, depth_or_img, holo_color_intrinsic, 0.10,
                        #                           tf_avg, color_or_img_rgb, model_cls, device, pre_process)

                im0 = annotator.result()

                # publish
                bbox_msg = self.cv_bridge.cv2_to_imgmsg(im0)
                bbox_msg.header.stamp = depth_frame_stamp
                bbox_msg.header.frame_id = depth_frame_id
                bbox_msg.encoding = "rgb8"
                self.box_pub.publish(bbox_msg)

                color_bbox_msg = self.cv_bridge.cv2_to_imgmsg(image_draw)
                color_bbox_msg.header.stamp = depth_frame_stamp
                color_bbox_msg.header.frame_id = depth_frame_id
                color_bbox_msg.encoding = "rgb8"
                self.color_box_pub.publish(color_bbox_msg)


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

print(ROOT)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="holo_ROS")
    parser.add_argument(
        "--save_path",
        dest="save_path",
        help="Path to save results",
        default="output/",
        type=str,
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[640],
        help="inference size h,w",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default=ROOT / "test/weights/1028_70.pt",
        help="model path(s)"
        # "--weights", nargs="+", type=str, default=ROOT / "best.pt", help="model path(s)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=ROOT / "data/depth_hand.yaml",
        help="(optional) dataset.yaml path",
    )
    parser.add_argument(
        "--device", default=1, type=str, help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--half", action="store_true", help="use FP16 half-precision inference"
    )
    parser.add_argument(
        "--dnn", action="store_true", help="use OpenCV DNN for ONNX inference"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # network.eval()
    args = parse_args()

    device = select_device(args.device)
    model = DetectMultiBackend(
        args.weights, device=device, dnn=args.dnn, data=args.data, fp16=args.half
    )
    print("Loading the Classification Model......")
    model_cls = torch.load(
        os.path.join(curr_dir, "./test/weights_cls/model_res18_1106_1_4.pt")
    )
    model_cls.eval()
    pre_process = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5484, 0.4518, 0.3903], [0.1798, 0.2289, 0.2124])
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    # cfg = get_cfg()
    # cfg.OUTPUT_DIR = os.path.join(curr_dir, "model_output")

    # config_name = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    # cfg.merge_from_file(model_zoo.get_config_file(config_name))
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_name)
    # cfg.DATASETS.TEST = ()
    # cfg.DATASETS.TRAIN = (os.path.join(curr_dir, "my_dataset"),)

    # cfg.DATALOADER.NUM_WORKERS = 1
    # cfg.SOLVER.IMS_PER_BATCH = 20
    # cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    # cfg.SOLVER.MAX_ITER = 10000
    # cfg.SOLVER.CHECKPOINT_PERIOD = (
    #     100000  # Small value=Frequent save need a lot of storage.
    # )
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 14

    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(0.3)

    # cfg.DATASETS.TEST = (os.path.join(curr_dir, "my_test_dataset"),)

    # predictor = DefaultPredictor(cfg)


    cfg = get_cfg()
    cfg.OUTPUT_DIR = os.path.join(curr_dir, "model_output")

    config_name = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_name))
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_1108.pth")
    cfg.DATASETS.TEST = ()
    cfg.DATASETS.TRAIN = (os.path.join(curr_dir, "my_dataset"),)

    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.SOLVER.IMS_PER_BATCH = 30
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.CHECKPOINT_PERIOD = (
        100000  # Small value=Frequent save need a lot of storage.
    )
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 42

   

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(0.3)

    cfg.DATASETS.TEST = (os.path.join(curr_dir, "my_test_dataset"),)

    predictor = DefaultPredictor(cfg)













    m = torch.nn.Softmax(dim=1)

    # image listener
    listener = ImageListener(
        topics=[
            "/hololens2/sensor_depth/image_raw",
            "/hololens2/sensor_color/image_raw",
            "/hololens2/sensor_color/camera_info",
        ]
    )
    index = 0
    while not rospy.is_shutdown():
        listener.run_network(
            model,
            m,
            index=index,
            model_cls=model_cls,
            model_obj=predictor,
            device=device,
            pre_process=pre_process,
            augment=args.augment,
        )
        index = index + 1
