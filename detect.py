from detector.yolo_detect import ObjectDetectionYolo
from detector.image_process_detect import ImageProcessDetection
from detector.yolo_asff_detector import ObjectDetectionASFF
from detector.visualize import BBoxVisualizer
from config import config
from utils.utils import gray3D, box2str
from utils.region_count import Region_count
import torch
import numpy as np
import cv2
import copy
<<<<<<< HEAD
=======
# video_path = "Video/origin/{}.mp4".format(config.video_num)
>>>>>>> 41f98f66debb27187072f5480fc4a14eba6a9926
write_box = False
import sys
print(sys.path)

class RegionDetector(object):
    def __init__(self, path):
        self.black_yolo = ObjectDetectionYolo(batchSize = 1, cfg="yolo/cfg/yolov3-spp-1cls.cfg", weights='models/yolo/best_converted.weights')
        self.gray_yolo = ObjectDetectionYolo(batchSize=1, cfg="yolo/cfg/prune_0.93_keep_0.1.cfg", weights='models/yolo/best.weights')

        self.BBV = BBoxVisualizer()
        self.dip_detection = ImageProcessDetection()
        self.cap = cv2.VideoCapture(path)
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=200, detectShadows=False)

        self.region_det = Region_count()
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
                    # black picture
                    enhance_kernel = np.array([[0, -1, 0], [0, 5, 0], [0, -1, 0]])
                    enhanced = cv2.filter2D(diff, -1, enhance_kernel)
                    inps, orig_img, black_boxes, scores, pt1, pt2 = self.black_yolo.process(enhanced)
                    if black_boxes is not None:
                        object_list, region = self.region_det.determine_within(enhanced.shape,black_boxes)
                        # count region number
                        for key, value in region.items():
                            print(region)
                            if key not in object_list:
                                if region[key] > 0:
                                    region[key] -= 1
                                else:
                                    region[key] = 0
                            else:
                                region[key] += 1
                                # only detect region which is processed
                                if region[key] > 400:
                                    cv2.putText(enhanced, 'HELP!!!!!!', (100, 200), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 0),5)
                    else:
                        object_list, region = self.region_det.determine_within(enhanced.shape, black_boxes)
                        for key, value in region.items():
                            if region[key] > 0:
                                region[key] -= 1

                    if black_boxes is not None:
                        enhanced = self.BBV.visualize_black(black_boxes, enhanced)
                        if write_box:
                            enhance_box = box2str(black_boxes.tolist())
                            self.black_file.write(enhance_box)
                            self.black_file.write("\n")
                    else:
                        if write_box:
                            self.black_file.write("\n")


                    # gray pics process
                    gray_img = gray3D(frame)
                    # gray_boxes, gray_cls, gray_score = self.gray_yolo.detect(gray_img)
                    inps, orig_img, gray_boxes, scores, pt1, pt2 = self.gray_yolo.process(gray_img)


                    # inps1, orig_img1, gray_boxes1, scores1, pt11, pt21 = self.gray_yolo1.process(gray_img)
                    # if gray_boxes1 is not None:
                    #     # self.region_det.determine_within(gray_img.shape, gray_boxes, gray_img)
                    #     gray_img1 = self.BBV.visualize_gray(gray_boxes1, gray_img)
                    #     if write_box:
                    #         gray_bbox = box2str(gray_boxes1)
                    #         self.gray_file.write(gray_bbox)
                    #         self.gray_file.write("\n")
                    # else:
                    #     if write_box:
                    #         self.gray_file.write("\n")
                    # gray_img1 = cv2.resize(gray_img1, (1000, 500))
                    # cv2.imshow("gray_result_train", gray_img1)

                    # print(gray_boxes)

                    if gray_boxes is not None:
                        # self.region_det.determine_within(gray_img.shape, gray_boxes, gray_img)
                        gray_img = self.BBV.visualize_gray(gray_boxes, gray_img)
                        if write_box:
                            gray_bbox = box2str(gray_boxes)
                            self.gray_file.write(gray_bbox)
                            self.gray_file.write("\n")
                    else:
                        if write_box:
                            self.gray_file.write("\n")


                # cv2.imshow("dip_result", dip_img)
                enhanced = cv2.resize(enhanced,(1000,500))
                self.region_det.drawlines(enhanced,enhanced.shape)
                cv2.imshow("black_result", enhanced)
                gray_img = cv2.resize(gray_img, (1000, 500))
                self.region_det.drawlines(gray_img, gray_img.shape)
                cv2.imshow("gray_result", gray_img)

                cnt += 1
                # print(cnt)
                cv2.waitKey(10)
            else:
                self.cap.release()
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    # for num in [4,5,19,18,38,46,48,53]:
    #     print("Processing video {}.avi".format(num))
    num = 4
    RD = RegionDetector("./Video/origin/{}.mp4".format(num))
    RD.process()
