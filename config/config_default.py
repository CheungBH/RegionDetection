device = "cuda:0"

black_yolo_cfg = "yolo/cfg/yolov3-spp-1cls.cfg"
black_yolo_weights = 'models/yolo/best_converted.weights'
gray_yolo_cfg = "yolo/cfg/prune_0.93_keep_0.1.cfg"
gray_yolo_weights = 'models/yolo/best.weights'

# For yolo
confidence = 0.05
num_classes = 80
nms_thresh = 0.33
input_size = 416

# For pose estimation
input_height = 320
input_width = 256
output_height = 80
output_width = 64
fast_inference = True
pose_batch = 80


water_top = 40

# For detection
video_num = 5
video_path = "video/origin/{}.mp4".format(video_num)
frame_size = (540, 360)

black_box_threshold = 0.3
gray_box_threshold = 0.3
