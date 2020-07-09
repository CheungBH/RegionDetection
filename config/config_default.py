# For yolo
confidence = 0.5
num_classes = 80
nms_thresh = 0.33
input_size = 608

# For pose estimation
input_height = 320
input_width = 256
output_height = 80
output_width = 64
fast_inference = True
pose_batch = 80


water_top = 40

# For detection
video_num = 48
video_path = "/media/hkuit164/7024-A194/0507 New"
frame_size = (720, 560)
rgb_yolo_cfg = "config/yolo_cfg/yolov3-spp-1cls.cfg"
rgb_yolo_weights = '/media/hkuit164/WD20EJRX/spp/150_16_dark_yes_608_multi/last.weights'
black_yolo_cfg = "config/yolo_cfg/yolov3-spp-1cls.cfg"
black_yolo_weights = '/media/hkuit164/WD20EJRX/black_we/200_16_dark_yes_416_multi_150/best.weights'
# black_yolo_weights = 'models/yolo/best_converted.weights'
gray_yolo_cfg = "config/yolo_cfg/yolov3-spp-1cls.cfg"
gray_yolo_weights = '/media/hkuit164/WD20EJRX/multi/gray_200_16_dark_yes_608_multi_135/best.weights'
# gray_yolo_weights ='/media/hkuit164/WD20EJRX/multi/200_16_dark_yes_416_multi_200_cls10/best.weights'
# gray_yolo_weights ='/media/hkuit164/WD20EJRX/11/200_16_dark_yes_416_multi_200_arc/last.weights'#best
# gray_yolo_weights = 'models/yolo/spp/200_16_dark_yes_608_multi/last.weights'#best
horizontal = 4
vertical = 3
device = 'cuda:0'

water_top = 40

# For detection
video_num = 5
video_path = "video/origin/{}.mp4".format(video_num)
frame_size = (540, 360)

black_box_threshold = 0.3
gray_box_threshold = 0.3
