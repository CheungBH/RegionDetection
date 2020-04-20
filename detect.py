from detector.yolo_detect import ObjectDetectionYolo
from detector.image_process_detect import ImageProcessDetection
from detector.yolo_asff_detector import ObjectDetectionASFF
from detector.visualize import BBoxVisualizer
from config import config
from utils.utils import gray3D, box2str
import torch
import numpy as np
import cv2
import copy


video_path = "Video/origin/{}.mp4".format(config.video_num)
write_box = True


class RegionDetector(object):
    def __init__(self, path):
        self.black_yolo = ObjectDetectionYolo()
        self.gray_yolo = ObjectDetectionASFF(cfg="yolo_asff/config/yolov3_baseline.cfg")
        self.BBV = BBoxVisualizer()
        self.dip_detection = ImageProcessDetection()
        self.cap = cv2.VideoCapture(path)
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=200, detectShadows=False)
        if write_box:
            self.black_file = open("Video/txt/black/{}.txt".format(path.split("/")[-1][:-4]), "w")
            self.gray_file = open("Video/txt/gray/{}.txt".format(path.split("/")[-1][:-4]), "w")

    def process(self):
        cnt = 0

        while True:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, config.frame_size)
                fgmask = self.fgbg.apply(frame)
                background = self.fgbg.getBackgroundImage()
                diff = cv2.absdiff(frame, background)
                dip_img = copy.deepcopy(frame)
                dip_box = self.dip_detection.detect_rect(diff)
                if dip_box:
                    dip_img = self.BBV.visualize_dip(dip_box, dip_img)

                with torch.no_grad():
                    enhance_kernel = np.array([[0, -1, 0], [0, 5, 0], [0, -1, 0]])
                    enhanced = cv2.filter2D(diff, -1, enhance_kernel)
                    inps, orig_img, black_boxes, scores, pt1, pt2 = self.black_yolo.process(enhanced)
                    if black_boxes is not None:
                        enhanced = self.BBV.visualize_black(black_boxes, enhanced)
                        if write_box:
                            enhance_box = box2str(black_boxes.tolist())
                            self.black_file.write(enhance_box)
                            self.black_file.write("\n")
                    else:
                        if write_box:
                            self.black_file.write("\n")

                    gray_img = gray3D(frame)
                    gray_boxes, gray_cls, gray_score = self.gray_yolo.detect(gray_img)
                    if gray_boxes is not None:
                        gray_img = self.BBV.visualize_gray(gray_boxes, gray_img)
                        if write_box:
                            gray_bbox = box2str(gray_boxes)
                            self.gray_file.write(gray_bbox)
                            self.gray_file.write("\n")
                    else:
                        if write_box:
                            self.gray_file.write("\n")

                cv2.imshow("dip_result", dip_img)
                cv2.imshow("black_result", enhanced)
                cv2.imshow("gray_result", gray_img)

                cnt += 1
                print(cnt)
                cv2.waitKey(10)
            else:
                self.cap.release()
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    for num in [4,5,19,38,46,48,53]:
        RD = RegionDetector(video_path)
        RD.process()
