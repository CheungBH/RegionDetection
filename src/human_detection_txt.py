import torch
try:
    import src.debug.config.cfg_only_detections as config
except:
    import config.config as config
import numpy as np
import cv2
import copy
from src.detector.yolo_detect import ObjectDetectionYolo
from src.detector.image_process_detect import ImageProcessDetection
# from src.detector.yolo_asff_detector import ObjectDetectionASFF
# For nano jetson
# from src.detector.yolo_trt_detect import TrtYOLO
from src.detector.visualize import BBoxVisualizer
from utils.utils import str2box, str2score
from src.utils.img import gray3D
from src.detector.box_postprocess import eliminate_nan, filter_box, BoxEnsemble
from src.tracker.track import ObjectTracker
from src.tracker.visualize import IDVisualizer
from src.analyser.area import RegionProcessor
from src.analyser.humans import HumanProcessor
from src.utils.utils import paste_box


fourcc = cv2.VideoWriter_fourcc(*'XVID')
empty_tensor = torch.empty([0, 7])
empty_tensor4 = torch.empty([0, 4])


class ImgProcessor:
    def __init__(self, path, resize_size, show_img=True):
        with open(path.replace(path.split("/")[-1], "black.txt"), "r") as bf:
            self.black_boxes = [line[:-1] for line in bf.readlines()]
        with open(path.replace(path.split("/")[-1], "gray.txt"), "r") as gf:
            self.gray_boxes = [line[:-1] for line in gf.readlines()]
        with open(path.replace(path.split("/")[-1], "black_score.txt"), "r") as bsf:
            self.black_scores = [line[:-1] for line in bsf.readlines()]
        with open(path.replace(path.split("/")[-1], "gray_score.txt"), "r") as gsf:
            self.gray_scores = [line[:-1] for line in gsf.readlines()]
        # For nano jetson
        # self.yolo_trt = TrtYOLO(model_path, (720, 540), 1)
        self.BBV = BBoxVisualizer()
        self.object_tracker = ObjectTracker()
        self.dip_detection = ImageProcessDetection()
        self.BBV = BBoxVisualizer()
        self.IDV = IDVisualizer()
        self.img = []
        self.id2bbox = {}
        self.img_black = []
        self.show_img = show_img
        self.RP = RegionProcessor(resize_size[0], resize_size[1], 10, 10)
        self.HP = HumanProcessor(resize_size[0], resize_size[1])
        self.BE = BoxEnsemble(resize_size[0], resize_size[1])
        self.resize_size = resize_size

    def init(self):
        # Initialize all the status
        self.RP = RegionProcessor(self.resize_size[0], self.resize_size[1], 10, 10)
        self.HP = HumanProcessor(self.resize_size[0], self.resize_size[1])
        self.object_tracker = ObjectTracker()
        self.object_tracker.init_tracker()

    def process_img(self, frame, background):
        # Initial the images
        rgb_kps, dip_img, track_pred, rd_box = \
            copy.deepcopy(frame), copy.deepcopy(frame), copy.deepcopy(frame), copy.deepcopy(frame)
        img_black = np.full((self.resize_size[1], self.resize_size[0], 3), 0).astype(np.uint8)
        iou_img, black_kps, img_size_ls, img_box_ratio, rd_cnt = copy.deepcopy(img_black), \
            copy.deepcopy(img_black), copy.deepcopy(img_black), copy.deepcopy(img_black), copy.deepcopy(img_black)

        [black_boxes, black_scores, gray_boxes, gray_scores, black_res, gray_res] = [empty_tensor] * 6
        diff = cv2.absdiff(frame, background)

        with torch.no_grad():
            # black image process
            enhance_kernel = np.array([[0, -1, 0], [0, 5, 0], [0, -1, 0]])
            enhanced = cv2.filter2D(diff, -1, enhance_kernel)
            black_boxes = str2box(self.black_boxes.pop(0))
            black_scores = str2score(self.black_scores.pop(0))
            if black_boxes is not None:
                black_boxes = torch.Tensor(black_boxes)
                black_scores = torch.Tensor(black_scores)
                black_res = torch.cat((black_boxes, black_scores, torch.ones(len(black_boxes),1),
                                       torch.zeros(len(black_boxes),1)), axis=1)
                # black_res = [black_boxes[idx] + [black_scores[idx]] + [0.999, 0] for idx in range(len(black_boxes))]
                self.BBV.visualize(black_boxes, enhanced, black_scores)
                black_boxes, black_scores, black_res = \
                    filter_box(black_boxes, black_scores, black_res, config.black_box_threshold)
            black_results = [enhanced, black_boxes, black_scores]

            # gray image process
            gray_img = gray3D(frame)
            gray_boxes = str2box(self.gray_boxes.pop(0))
            gray_scores = str2score(self.gray_scores.pop(0))
            if gray_boxes is not None:
                gray_boxes = torch.Tensor(gray_boxes)
                gray_scores = torch.Tensor(gray_scores)
                gray_res = torch.cat((gray_boxes, gray_scores, torch.ones(len(gray_boxes),1),
                                      torch.zeros(len(gray_boxes),1)), axis=1)
                #gray_res = [gray_boxes[idx] + [gray_scores[idx]] + [0.999, 0] for idx in range(len(gray_boxes))]
                self.BBV.visualize(gray_boxes, gray_img, gray_scores)
                gray_boxes, gray_scores, gray_res = \
                    filter_box(gray_boxes, gray_scores, gray_res, config.gray_box_threshold)
            gray_results = [gray_img, gray_boxes, gray_scores]

            # merge gray and black image
            merged_res = self.BE.ensemble_box(black_res, gray_res)

            # tracking
            '''
            self.id2bbox is the interface
            '''
            self.id2bbox = self.object_tracker.track(merged_res)
            self.id2bbox = eliminate_nan(self.id2bbox)
            boxes = self.object_tracker.id_and_box(self.id2bbox)
            self.IDV.plot_bbox_id(self.id2bbox, track_pred, color=("red", "purple"), with_bbox=True)
            self.IDV.plot_bbox_id(self.object_tracker.get_pred(), track_pred, color=("yellow", "orange"), id_pos="down",
                                  with_bbox=True)

            # draw tracking images
            self.object_tracker.plot_iou_map(iou_img)
            img_box_ratio = paste_box(rgb_kps, img_box_ratio, boxes)
            self.HP.update(self.id2bbox)

            # Region process
            self.RP.process_box(boxes, rd_box, rd_cnt)
            warning_idx = self.RP.get_alarmed_box_id(self.id2bbox)

            # h-w ratio visualize
            self.HP.vis_box_size(img_box_ratio, img_size_ls)

            # Concat the box
            detection_map = np.concatenate((enhanced, gray_img), axis=1)
            tracking_map = np.concatenate((track_pred, iou_img), axis=1)
            row_1st_map = np.concatenate((detection_map, tracking_map), axis=1)
            box_map = np.concatenate((img_box_ratio, img_size_ls), axis=1)
            rd_map = np.concatenate((rd_cnt, rd_box), axis=1)
            row_2nd_map = np.concatenate((rd_map, box_map), axis=1)
            res = np.concatenate((row_1st_map, row_2nd_map), axis=0)

        return gray_results, black_results, 0, res
