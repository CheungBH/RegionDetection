device = "cuda:0"

gray_yolo_cfg = "weights/yolo/finetuned/0909_test-test1-best-ALL-prune_0.98_keep_0.01_18_shortcut/prune_0.98_keep_0.01_18_shortcut.cfg"
gray_yolo_weights = "weights/yolo/finetuned/0909_test-test1-best-ALL-prune_0.98_keep_0.01_18_shortcut/best.weights"
black_yolo_cfg = "weights/yolo/0710/black/yolov3-spp-1cls.cfg"
black_yolo_weights = "weights/yolo/0710/black/150_416_best.weights"
rgb_yolo_cfg = ""
rgb_yolo_weights = ""

pose_weight = "weights/sppe/duc_se.pth"
pose_cfg = None

video_path = "video/0507_mul_01.mp4"
water_top = 40

RNN_frame_length = 4
RNN_backbone = "TCN"
RNN_class = ["stand", "drown"]
RNN_weight = "weights/RNN/TCN_struct1_2020-07-08-20-02-32.pth"
TCN_single = True

'''
----------------------------------------------------------------------------------------------------------------
'''

# For yolo
confidence = 0.4
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

pose_backbone = "seresnet101"
pose_cls = 17
DUCs = [480, 240]


# For detection
frame_size = (720, 540)
resize_ratio = 0.5
store_size = (frame_size[0]*4, frame_size[1]*2)
show_size = (1560, 720)

black_box_threshold = 0.3
gray_box_threshold = 0.2

