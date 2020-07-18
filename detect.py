import cv2
from config import config
from src.human_detection import ImgProcessor
import numpy as np
from utils.utils import write_file
import os
write_box = False
write_video = True
fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame_size = config.frame_size


class RegionDetector(object):
    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=200, detectShadows=False)
        self.IP = ImgProcessor()
        if write_box:
            self.black_file = open("video/txt/black/{}.txt".format(path.split("/")[-1][:-4]), "w")
            self.gray_file = open("video/txt/gray/{}.txt".format(path.split("/")[-1][:-4]), "w")
            self.black_score_file = open("video/txt/black_score/{}.txt".format(path.split("/")[-1][:-4]), "w")
            self.gray_score_file = open("video/txt/gray_score/{}.txt".format(path.split("/")[-1][:-4]), "w")
            # self.out_video = cv2.VideoWriter("video/processed/" + path.split("/")[-1], fourcc, 15,
            #                                  (frame_size[0]*2, frame_size[1]))
        if write_video:
            self.out_video = cv2.VideoWriter("output.mp4", fourcc, 15, (frame_size[0]*2, frame_size[1]))

    def process(self):
        cnt = 0
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter('/media/hkuit164/WD20EJRX/Begin with people/video0507/{}'.format(self.path.split('/')[-2]+self.path.split('/')[-1]), fourcc, 20.0, (1440,560))
        while True:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, config.frame_size)
                fgmask = self.fgbg.apply(frame)
                background = self.fgbg.getBackgroundImage()

                gray_res, black_res, dip_res, rd_map = self.IP.process_img(frame, background)

                if write_box:
                    write_file(gray_res, self.gray_file, self.gray_score_file)
                    write_file(black_res, self.black_file, self.black_score_file)

                # dip_img = cv2.resize(dip_res[0], frame_size)
                # cv2.imshow("dip_result", dip_img)
                enhanced = cv2.resize(black_res[0], frame_size)
                # cv2.imshow("black_result", enhanced)
                gray_img = cv2.resize(gray_res[0], frame_size)
                # cv2.imshow("gray_result", g1ray_img)

                yolo_map = np.concatenate((enhanced, gray_img), axis=1)
                res = np.concatenate((yolo_map, rd_map), axis=0)
                cv2.resize(res, (1440, 720))
                cv2.imshow("res", res)

                if write_video:
                    self.out_video.write(res)
                # out.write(res)
                cnt += 1
                cv2.waitKey(1)
            else:
                self.cap.release()
                self.out_video.release()
                cv2.destroyAllWindows()
                self.IP.RP.out.release()
                break


if __name__ == '__main__':
    # for path in os.listdir(config.video_path):
    #     for name in os.listdir(config.video_path+'/'+path):
    #         aa = config.video_path+'/'+path+'/'+name
    #         print(aa)
    RD = RegionDetector(config.video_path)
    RD.process()

    # import shutil
    # src = "video/619_Big Group"
    # for folder in os.listdir(src):
    #     video_folder = os.path.join(src, folder)
    #     dest_folder = video_folder + "_processed"
    #     os.makedirs(dest_folder, exist_ok=True)
    #
    #     for v_name in os.listdir(video_folder):
    #         video = os.path.join(video_folder, v_name)
    #         RD = RegionDetector(video)
    #         RD.process()
    #
    #         shutil.copy("output2.mp4", os.path.join(dest_folder, "rd_" + v_name))
    #         shutil.copy("output.mp4", os.path.join(dest_folder, "yolo_" + v_name))
