device = "cuda:0"

gray_yolo_cfg = "model/gray/prune_0.93_keep_0.1.cfg"
gray_yolo_weights = "model/gray/best.weights"
black_yolo_cfg = "model/black/yolov3-spp-1cls.cfg"
black_yolo_weights = "model/black/best_converted.weights"
rgb_yolo_cfg = ""
rgb_yolo_weights = ""

video_path = "video/underwater/multiple.mp4"
water_top = 40

'''
----------------------------------------------------------------------------------------------------------------
'''

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


# For detection
frame_size = (720, 540)
store_size = (frame_size[0]*2, frame_size[1]*2)

black_box_threshold = 0.3
gray_box_threshold = 0.3
