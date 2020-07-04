from src.detector.yolo_detect import ObjectDetectionYolo
from src.detector.image_process_detect import ImageProcessDetection
# from src.detector.yolo_asff_detector import ObjectDetectionASFF
from src.detector.visualize import BBoxVisualizer
from config import config
from utils.utils import gray3D, box2str
from utils.region_count import Region_count
import torch
import numpy as np
import cv2
import copy
from config import config
from src.human_detection import ImgProcessor

write_box = False
frame_size = config.frame_size

IP = ImgProcessor()


class RegionDetector(object):
    def __init__(self, path):

        self.cap = cv2.VideoCapture(path)
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=200, detectShadows=False)

    def process(self):
        cnt = 0
        while True:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, config.frame_size)
                fgmask = self.fgbg.apply(frame)
                background = self.fgbg.getBackgroundImage()

                gray_res, black_res, dip_res = IP.process_img(frame, background)

                # dip_img = cv2.resize(dip_res[0], frame_size)
                # cv2.imshow("dip_result", dip_img)
                enhanced = cv2.resize(black_res[0], frame_size)
                cv2.imshow("black_result", enhanced)
                gray_img = cv2.resize(gray_res[0], frame_size)
                cv2.imshow("gray_result", gray_img)

                cnt += 1
                cv2.waitKey(10)
            else:
                self.cap.release()
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    RD = RegionDetector(config.video_path)
    RD.process()
