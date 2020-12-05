import torch
import os


"-------------------Outer configuration-----------------------"

video_path = "video/sample/0048.mp4"
gray_yolo_cfg = "weights/yolo/finetuned/0909_test-test1-best-ALL-prune_0.96_keep_0.01_15_" \
                "shortcut/prune_0.96_keep_0.01_15_shortcut.cfg"
gray_yolo_weights = "weights/yolo/finetuned/0909_test-test1-best-ALL-prune_0.96_keep_0.01_15_shortcut/best.weights"
black_yolo_cfg = "weights/yolo/black/yolov3-spp-1cls.cfg"
black_yolo_weights = "weights/yolo/black/best_converted.weights"
black_box_threshold = 0.5122
gray_box_threshold = 0.526

img_folder = "img/squat"
water_top = 40

CNN_weight = "model/CNN/underwater/1/1_mobilenet_9_decay1.pth"

write_video = False
write_box = True
write_kps = False

resize_ratio = 0.5
show_size = (1280, 640)
store_size = (3840, 2160)


"-------------------Inner configuration-----------------------"

device = "cuda:0"
print("Using {}".format(device))

confidence = 0.8
num_classes = 80
nms_thresh = 0.33
input_size = 416


track_idx = "all"    # If all idx, track_idx = "all"
track_plot_id = ["all"]   # If all idx, track_plot_id = ["all"]
assert track_idx == "all" or isinstance(track_idx, int)

plot_bbox = True
plot_kps = True
plot_id = True

libtorch = False


"----------------------------Set inner configuration with opt-------------------------"

from src.opt import opt

opt.device = device

opt.confidence = confidence
opt.num_classes = num_classes
opt.nms_thresh = nms_thresh
opt.input_size = input_size

opt.libtorch = libtorch

opt.plot_bbox = plot_bbox
opt.plot_kps = plot_kps
opt.plot_id = plot_id
opt.track_id = track_idx
opt.track_plot_id = track_plot_id

