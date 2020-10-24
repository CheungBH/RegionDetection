import cv2
from config import config
from src.human_detection import ImgProcessor
import numpy as np
from utils.utils import write_file


store_size = config.store_size
resize_ratio = config.resize_ratio
show_size = config.show_size


from threading import Thread
import time
from queue import Queue
#https://github.com/Kjue/python-opencv-gpu-video


class RegionDetectorThread:
    def __init__(self, path, queueSize=3000):
        self.path = path
        # self.video = UMatFileVideoStream(self.path, 128).start()
        # self.rgb = cv2.UMat(self.height, self.width, cv2.CV_8UC3)
        self.cap = cv2.VideoCapture(path)
        self.stopped = False
        # # initialize the queue used to store frames read from
        # # the video file
        self.Q = Queue(maxsize=queueSize)
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=200, detectShadows=False)
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.resize_size = (int(self.width * resize_ratio), int(self.height * resize_ratio))
        self.IP = ImgProcessor(self.resize_size)

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def update(self):
        # keep looping infinitely
        self.IP.init()
        # IP.object_tracker.init_tracker()
        cnt = 0
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return
            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.cap.read()
                start = time.time()
                if grabbed:
                    frame = cv2.resize(frame, self.resize_size)
                    fgmask = self.fgbg.apply(frame)
                    background = self.fgbg.getBackgroundImage()
                    gray_res, black_res, dip_res, res_map = self.IP.process_img(frame, background)

                    cv2.imshow("res", cv2.resize(res_map, show_size))
                    # out.write(res)
                    cnt += 1
                    cv2.waitKey(1)
                    all_time = time.time() - start
                    print("time is:", all_time)
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stop()
                    return
                # add the frame to the queue
                self.Q.put(frame)
            else:
                self.Q.queue.clear()


if __name__ == '__main__':
    # RD = RegionDetector(config.video_path)
    # RD.process()
    RegionDetectorThread(config.video_path).update()
