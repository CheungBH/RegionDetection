from detector.yolo_detect import ObjectDetectionYolo
from detector.image_process_detect import ImageProcessDetection
from detector.yolo_asff_detector import ObjectDetectionASFF
from detector.visualize import BBoxVisualizer
from config import config
from utils.utils import gray3D, box2str, score2str
import torch
import numpy as np
import cv2
import copy


# video_path = "Video/origin/{}.mp4".format(config.video_num)
write_box = True
fourcc = cv2.VideoWriter_fourcc(*'XVID')


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
            self.black_score_file = open("Video/txt/black_score/{}.txt".format(path.split("/")[-1][:-4]), "w")
            self.gray_score_file = open("Video/txt/gray_score/{}.txt".format(path.split("/")[-1][:-4]), "w")
            self.out_video = cv2.VideoWriter("Video/processed/" + path.split("/")[-1], fourcc, 15, (config.frame_size[0]*2, config.frame_size[1]))

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
                    inps, orig_img, black_boxes, black_scores, pt1, pt2 = self.black_yolo.process(enhanced)
                    if black_boxes is not None:
                        enhanced = self.BBV.visualize_black(black_boxes, enhanced, black_scores.squeeze().tolist())
                        if write_box:
                            enhance_box = box2str(black_boxes.tolist())
                            black_score = score2str(black_scores.squeeze().tolist())
                            self.black_file.write(enhance_box)
                            self.black_file.write("\n")
                            self.black_score_file.write(black_score)
                            self.black_score_file.write("\n")
                    else:
                        if write_box:
                            self.black_file.write("\n")
                            self.black_score_file.write("\n")

                    gray_img = gray3D(frame)
                    gray_boxes, gray_cls, gray_scores = self.gray_yolo.detect(gray_img)
                    if gray_boxes is not None:
                        gray_img = self.BBV.visualize_gray(gray_boxes, gray_img, gray_scores)
                        if write_box:
                            gray_bbox = box2str(gray_boxes)
                            gray_score = score2str(gray_scores)
                            self.gray_file.write(gray_bbox)
                            self.gray_file.write("\n")
                            self.gray_score_file.write(gray_score)
                            self.gray_score_file.write("\n")
                    else:
                        if write_box:
                            self.gray_file.write("\n")
                            self.gray_score_file.write("\n")

                res = np.concatenate((enhanced, gray_img), axis=1)
                cv2.imshow("res", res)
                if write_box:
                    self.out_video.write(res)
                # cv2.imshow("dip_result", dip_img)
                # cv2.imshow("black_result", enhanced)
                # cv2.imshow("gray_result", gray_img)

                cnt += 1
                print(cnt)
                cv2.waitKey(10)
            else:
                self.cap.release()
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    for num in [4,5,18,19,38,46,53,48, "02", "01", "03", "04", "05", "06", "07", "08", "09", "010", '011']:
    # for num in [4]:
        print("Processing video {}.avi".format(num))
        RD = RegionDetector("Video/origin/{}.mp4".format(num))
        RD.process()
    # for num in ["10", "11"]:
    #     v_num = "0" + str(num)
    #     video_name = "./Video/origin/{}.mp4".format(v_num)
    #     # print("Processing video {}.avi".format(num))
    #     # num = 4
    #     RD = RegionDetector(video_name)
    #     RD.process()
