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
video_num = 48
video_path = "Video/origin/{}.mp4".format(video_num)
frame_size = (540, 360)