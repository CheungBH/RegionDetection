import os
import cv2
from config import config

src_folder = "../Video/origin"
dest_folder = src_folder + "_lr"
os.makedirs(dest_folder, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

for video_name in os.listdir(src_folder):
    print("Processing video {}".format(video_name))
    cap = cv2.VideoCapture(os.path.join(src_folder, video_name))
    out_video = cv2.VideoWriter(os.path.join(dest_folder,video_name), fourcc, 15, config.frame_size)

    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, config.frame_size)
            out_video.write(frame)
        else:
            out_video.release()
            cap.release()
            break

