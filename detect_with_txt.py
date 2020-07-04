from src.detector.image_process_detect import ImageProcessDetection
from src.detector.visualize import BBoxVisualizer
from config import config
from utils.utils import str2box, str2score
from src.utils.img import gray3D
import torch
import numpy as np
import cv2


video_path = "Video/origin/{}.mp4".format(config.video_num)


class DrownDetector(object):
    def __init__(self, path):
        self.BBV = BBoxVisualizer()
        self.dip_detection = ImageProcessDetection()
        self.cap = cv2.VideoCapture(path)
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=200, detectShadows=False)
        with open("Video/txt/black/{}.txt".format(path.split("/")[-1][:-4]), "r") as bf:
            self.black_boxes = [line[:-1] for line in bf.readlines()]
        with open("Video/txt/gray/{}.txt".format(path.split("/")[-1][:-4]), "r") as gf:
            self.gray_boxes = [line[:-1] for line in gf.readlines()]
        with open("Video/txt/black_score/{}.txt".format(path.split("/")[-1][:-4]), "r") as bsf:
            self.black_scores = [line[:-1] for line in bsf.readlines()]
        with open("Video/txt/gray_score/{}.txt".format(path.split("/")[-1][:-4]), "r") as gsf:
            self.gray_scores = [line[:-1] for line in gsf.readlines()]
        self.cap_processed = cv2.VideoCapture(path.replace("origin", "processed"))

    def process(self):
        cnt = 0

        while True:
            ret, frame = self.cap.read()
            ret2, processed_frame = self.cap_processed.read()
            if ret:
                frame = cv2.resize(frame, config.frame_size)
                gray_frame = gray3D(frame)
                fgmask = self.fgbg.apply(frame)
                background = self.fgbg.getBackgroundImage()
                diff = cv2.absdiff(frame, background)
                # dip_img = copy.deepcopy(frame)
                # dip_box = self.dip_detection.detect_rect(diff)
                #
                # if dip_box:
                #     dip_img = self.BBV.visualize(dip_box, dip_img)

                with torch.no_grad():
                    enhance_kernel = np.array([[0, -1, 0], [0, 5, 0], [0, -1, 0]])
                    enhanced = cv2.filter2D(diff, -1, enhance_kernel)
                    black_boxes = str2box(self.black_boxes.pop(0))
                    black_scores = str2score(self.black_scores.pop(0))
                    if black_boxes is not None:
                        # enhanced = self.BBV.visualize_black(black_boxes, enhanced, black_scores)
                        # filtered = [(b,s) for idx, (b, s) in enumerate(zip(black_boxes, black_scores)) if s > config.black_box_threshold]
                        filtered_black_boxes, filtered_black_scores = [], []
                        for b,s in zip(black_boxes, black_scores):
                            if s > config.black_box_threshold:
                                filtered_black_boxes.append(b)
                                filtered_black_scores.append(s)
                        enhanced = self.BBV.visualize(filtered_black_boxes, enhanced, filtered_black_scores)

                    gray_boxes = str2box(self.gray_boxes.pop(0))
                    gray_scores = str2score(self.gray_scores.pop(0))
                    if gray_boxes is not None:
                        # gray_frame = self.BBV.visualize_gray(gray_boxes, gray_frame, gray_scores)
                        filtered_gray_boxes, filtered_gray_scores = [], []
                        for b, s in zip(gray_boxes, gray_scores):
                            if s > config.gray_box_threshold:
                                filtered_gray_boxes.append(b)
                                filtered_gray_scores.append(s)
                        gray_frame = self.BBV.visualize(filtered_gray_boxes, gray_frame, filtered_gray_scores)

                # cv2.imshow("dip_result", dip_img)
                # cv2.moveWindow("dip_result", 0, 200)
                cv2.imshow("black_result", enhanced)
                # cv2.moveWindow("black_result", 550, 200)
                cv2.imshow("gray_result", gray_frame)
                # cv2.moveWindow("gray_result", 1100, 200)
                cv2.imshow("processed", processed_frame)

                # boxes, scores = self.BM.merge(gray_boxes, black_boxes, gray_scores, black_scores)
                #
                # if boxes is not None:
                #     inps, pt1, pt2 = crop_bbox(frame, torch.FloatTensor(boxes), torch.FloatTensor(scores))
                #     frame = self.BBV.visualize_gray(boxes, frame, scores)
                #     cv2.imshow("frame", frame)
                #     cv2.imshow("cropped", (torch_to_im(inps[0]) * 255))

                cnt += 1
                print(cnt)
                cv2.waitKey(50)
            else:
                self.cap.release()
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    DD = DrownDetector(video_path)
    DD.process()
