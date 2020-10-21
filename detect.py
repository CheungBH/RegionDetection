import cv2
from config import config
from src.human_detection import ImgProcessor
import numpy as np
from utils.utils import write_file

write_box = False
write_video = False

frame_size = config.frame_size
store_size = config.store_size
resize_ratio = config.resize_ratio
show_size = config.show_size


class RegionDetector(object):
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=200, detectShadows=False)
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.resize_size = (int(self.width * resize_ratio), int(self.height * resize_ratio))
        self.IP = ImgProcessor(self.resize_size)
        if write_box:
            self.black_file = open("video/txt/black/{}.txt".format(path.split("/")[-1][:-4]), "w")
            self.gray_file = open("video/txt/gray/{}.txt".format(path.split("/")[-1][:-4]), "w")
            self.black_score_file = open("video/txt/black_score/{}.txt".format(path.split("/")[-1][:-4]), "w")
            self.gray_score_file = open("video/txt/gray_score/{}.txt".format(path.split("/")[-1][:-4]), "w")

        if write_video:
            self.out_video = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'XVID'), 15, store_size)

    def process(self):
        self.IP.init()
        cnt = 0
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        while True:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, self.resize_size)
                fgmask = self.fgbg.apply(frame)
                background = self.fgbg.getBackgroundImage()
                gray_res, black_res, dip_res, res_map = self.IP.process_img(frame, background)

                if write_box:
                    write_file(gray_res, self.gray_file, self.gray_score_file)
                    write_file(black_res, self.black_file, self.black_score_file)

                if write_video:
                    self.out_video.write(res_map)

                cv2.imshow("res", cv2.resize(res_map, show_size))
                # out.write(res)
                cnt += 1
                cv2.waitKey(1)
            else:
                self.cap.release()
                # self.out_video.release()
                cv2.destroyAllWindows()
                # self.IP.RP.out.release()
                break


if __name__ == '__main__':
    # for path in os.listdir(config.video_path):
    #     for name in os.listdir(config.video_path+'/'+path):
    #         aa = config.video_path+'/'+path+'/'+name
    #         print(aa)
    RD = RegionDetector(config.video_path)
    RD.process()

    # import shutil
    # import os
    # # src = "video/619_Big Group"
    # # for folder in os.listdir(src):
    # # video_folder = os.path.join(src, folder)
    # video_folder = "D:/0619_bad"
    # dest_folder = video_folder + "_res"
    # os.makedirs(dest_folder, exist_ok=True)
    #
    # for v_name in os.listdir(video_folder):
    #     video = os.path.join(video_folder, v_name)
    #     RD = RegionDetector(video)
    #     RD.process()
    #
    #     # shutil.copy("output2.mp4", os.path.join(dest_folder, "rd_" + v_name))
    #     shutil.move("output.mp4", os.path.join(dest_folder, v_name))
