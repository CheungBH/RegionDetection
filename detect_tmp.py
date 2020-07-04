import cv2
from config import config
from src.human_detection import ImgProcessor
import numpy as np
from utils.utils import box2str, score2str, write_file

write_box = True
fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame_size = config.frame_size
IP = ImgProcessor()


class RegionDetector(object):
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=200, detectShadows=False)
        if write_box:
            self.black_file = open("Video/txt/black/{}.txt".format(path.split("/")[-1][:-4]), "w")
            self.gray_file = open("Video/txt/gray/{}.txt".format(path.split("/")[-1][:-4]), "w")
            self.black_score_file = open("Video/txt/black_score/{}.txt".format(path.split("/")[-1][:-4]), "w")
            self.gray_score_file = open("Video/txt/gray_score/{}.txt".format(path.split("/")[-1][:-4]), "w")
            self.out_video = cv2.VideoWriter("Video/processed/" + path.split("/")[-1], fourcc, 15,
                                             (frame_size[0]*2, frame_size[1]))

    # def __write_box(self, black, gray):
    #     (_, gray_boxes, gray_scores) = gray
    #     if gray_boxes is not None:
    #         gray_bbox = box2str(gray_boxes.tolist())
    #         gray_score = score2str(gray_scores.squeeze().tolist())
    #         self.gray_file.write(gray_bbox)
    #         self.gray_file.write("\n")
    #         self.gray_score_file.write(gray_score)
    #         self.gray_score_file.write("\n")
    #     else:
    #         self.gray_file.write("\n")
    #         self.gray_score_file.write("\n")
    #
    #     (_, black_boxes, black_scores) = black
    #     if black_boxes is not None:
    #         black_box = box2str(black_boxes.tolist())
    #         black_score = score2str(black_scores.squeeze().tolist())
    #         self.black_file.write(black_box)
    #         self.black_file.write("\n")
    #         self.black_score_file.write(black_score)
    #         self.black_score_file.write("\n")
    #     else:
    #         self.black_file.write("\n")
    #         self.black_score_file.write("\n")

    def process(self):
        cnt = 0
        while True:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, config.frame_size)
                fgmask = self.fgbg.apply(frame)
                background = self.fgbg.getBackgroundImage()

                gray_res, black_res, dip_res = IP.process_img(frame, background)

                if write_box:
                    write_file(gray_res, self.gray_file, self.gray_score_file)
                    write_file(black_res, self.black_file, self.black_score_file)

                # dip_img = cv2.resize(dip_res[0], frame_size)
                # cv2.imshow("dip_result", dip_img)
                enhanced = cv2.resize(black_res[0], frame_size)
                # cv2.imshow("black_result", enhanced)
                gray_img = cv2.resize(gray_res[0], frame_size)
                # cv2.imshow("gray_result", gray_img)

                res = np.concatenate((enhanced, gray_img), axis=1)
                cv2.imshow("res", res)

                if write_box:
                    self.out_video.write(res)

                cnt += 1
                cv2.waitKey(10)
            else:
                self.cap.release()
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    RD = RegionDetector(config.video_path)
    RD.process()
