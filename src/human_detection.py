import torch
import numpy as np
import cv2
import copy
from config import config
from src.detector.yolo_detect import ObjectDetectionYolo
from src.detector.image_process_detect import ImageProcessDetection
# from src.detector.yolo_asff_detector import ObjectDetectionASFF
from src.detector.visualize import BBoxVisualizer
from src.estimator.pose_estimator import PoseEstimator
from src.estimator.visualize import KeyPointVisualizer
from src.utils.img import gray3D
from src.detector.box_postprocess import crop_bbox, merge_box, filter_box
from src.tracker.track import ObjectTracker
from src.tracker.visualize import IDVisualizer
from src.analyser.area import RegionProcessor
from src.analyser.humans import HumanProcessor
from src.utils.utils import paste_box
from src.RNNclassifier.classify import RNNInference

try:
    from config.config import gray_yolo_cfg, gray_yolo_weights, black_yolo_cfg, black_yolo_weights, video_path
except:
    from src.debug.config.cfg_only_detections import gray_yolo_cfg, gray_yolo_weights, black_yolo_cfg, black_yolo_weights, video_path

fourcc = cv2.VideoWriter_fourcc(*'XVID')


class ImgProcessor:
    def __init__(self, show_img=True):
        self.black_yolo = ObjectDetectionYolo(cfg=black_yolo_cfg, weight=black_yolo_weights)
        self.gray_yolo = ObjectDetectionYolo(cfg=gray_yolo_cfg, weight=gray_yolo_weights)
        self.object_tracker = ObjectTracker()
        self.dip_detection = ImageProcessDetection()
        self.RNN_model = RNNInference()
        self.pose_estimator = PoseEstimator(pose_cfg=config.pose_cfg, pose_weight=config.pose_weight)
        self.KPV = KeyPointVisualizer()
        self.BBV = BBoxVisualizer()
        self.IDV = IDVisualizer(with_bbox=False)
        self.img = []
        self.id2bbox = {}
        self.img_black = []
        self.show_img = show_img
        self.RP = RegionProcessor(config.frame_size[0], config.frame_size[1], 10, 10)
        self.HP = HumanProcessor(config.frame_size[0], config.frame_size[1])
        self.kps = {}
        self.kps_score = {}

    def process_img(self, frame, background):
        black_boxes, black_scores, gray_boxes, gray_scores = None, None, None, None
        # rgb_kps = copy.deepcopy(frame)
        diff = cv2.absdiff(frame, background)
        dip_img = copy.deepcopy(frame)
        dip_boxes = self.dip_detection.detect_rect(diff)
        # if len(dip_boxes) > 0:
        #     dip_img = self.BBV.visualize(dip_boxes, dip_img)
        dip_results = [dip_img, dip_boxes]

        with torch.no_grad():
            # black picture
            enhance_kernel = np.array([[0, -1, 0], [0, 5, 0], [0, -1, 0]])
            enhanced = cv2.filter2D(diff, -1, enhance_kernel)
            black_res = self.black_yolo.process(enhanced)
            if black_res is not None:
                black_boxes, black_scores = self.black_yolo.cut_box_score(black_res)
                enhanced = self.BBV.visualize(black_boxes, enhanced, black_scores)
                black_boxes, black_scores, black_res = \
                    filter_box(black_boxes, black_scores, black_res, config.black_box_threshold)
            black_results = [enhanced, black_boxes, black_scores]

            # gray pics process
            gray_img = gray3D(frame)
            gray_res = self.gray_yolo.process(gray_img)
            if gray_res is not None:
                gray_boxes, gray_scores = self.gray_yolo.cut_box_score(gray_res)
                gray_img = self.BBV.visualize(gray_boxes, gray_img, gray_scores)
                gray_boxes, gray_scores, gray_res = \
                    filter_box(gray_boxes, gray_scores, gray_res, config.gray_box_threshold)

            gray_results = [gray_img, gray_boxes, gray_scores]

            img_black = cv2.imread("src/black.jpg")
            img_black = cv2.resize(img_black, config.frame_size)
            tmp = copy.deepcopy(img_black)
            black_kps = copy.deepcopy(img_black)

            if gray_res is not None:
                self.id2bbox = self.object_tracker.track(gray_res)
                boxes = self.object_tracker.id_and_box(self.id2bbox)
                frame = self.IDV.plot_bbox_id(self.id2bbox, frame)
            else:
                boxes = None

            rgb_kps = copy.deepcopy(frame)
            img_black = paste_box(rgb_kps, img_black, boxes)
            self.HP.update(self.id2bbox)

            rd_map = self.RP.process_box(boxes, frame)
            warning_idx = self.RP.get_alarmed_box_id(self.id2bbox)
            danger_idx = self.HP.box_size_warning(warning_idx)

            if danger_idx:
                danger_box = [v.numpy() for k, v in self.id2bbox.items() if k in danger_idx]
                danger_box = torch.FloatTensor(danger_box)
                inps, pt1, pt2 = crop_bbox(rgb_kps, danger_box)
                if inps is not None:
                    kps, kps_score, kps_id = self.pose_estimator.process_img(inps, danger_box, pt1, pt2)
                    if self.kps is not []:
                        self.kps, self.kps_score = self.object_tracker.match_kps(kps_id, kps, kps_score)
                        self.HP.update_kps(self.kps)
                        rgb_kps = self.KPV.vis_ske(rgb_kps, kps, kps_score)
                        rgb_kps = self.IDV.plot_skeleton_id(self.kps, rgb_kps)
                        rgb_kps = self.BBV.visualize(danger_box, rgb_kps)
                        black_kps = self.KPV.vis_ske_black(black_kps, kps, kps_score)
                        black_kps = self.IDV.plot_skeleton_id(self.kps, black_kps)

                        for n, idx in enumerate(self.kps.keys()):
                            if self.HP.if_enough_kps(idx):
                                RNN_res = self.RNN_model.predict_action(self.HP.obtain_kps(idx))
                                self.HP.update_RNN(idx, RNN_res)
                                self.RNN_model.vis_RNN_res(n, idx, self.HP.get_RNN_preds(idx), black_kps)
                                # print("Prediction of idx {}： {}".format(idx, RNN_res))

            cv2.putText(tmp, "TBC...", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
            box_map = self.HP.vis_box_size(img_black)
            detection_map = np.concatenate((enhanced, gray_img), axis=1)
            yolo_cnt_map = np.concatenate((detection_map, rd_map), axis=0)
            yolo_map = np.concatenate((yolo_cnt_map, box_map), axis=1)
            kps_img = np.concatenate((tmp, rgb_kps, black_kps), axis=1)
            res = np.concatenate((yolo_map, kps_img), axis=0)

        return gray_results, black_results, dip_results, res
