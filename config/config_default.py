device = "cuda:0"

black_yolo_cfg = "weights/yolo/0710/black/yolov3-spp-1cls.cfg"
black_yolo_weights = 'weights/yolo/0710/black/150_416_best.weights'
gray_yolo_cfg = "weights/yolo/0710/gray/yolov3-spp-1cls.cfg"
gray_yolo_weights = 'weights/yolo/0710/gray/135_608_best.weights'

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

video_path = "video/0710/0048.mp4"
frame_size = (720, 540)

black_box_threshold = 0.3
gray_box_threshold = 0.3
