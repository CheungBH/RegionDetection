<<<<<<< HEAD
import cv2
class Region_count:
    def __init__(self):
        self.horizontal = 3
        self.vertical = 4
=======
import collections
import cv2
class Region_count:
    def __init__(self):
        self.horizontal = 4
        self.vertical = 5
>>>>>>> master
        self.num = self.horizontal * self.vertical
        self.region = {n: 0 for n in range(self.num)}
        self.region_num = []
        for i in range(self.horizontal):
            for j in range(self.vertical):
                self.region_num.append((i,j))
        self.value_count = [n for n in range(self.num)]
        self.coor_region = dict(zip(self.region_num, self.value_count))
<<<<<<< HEAD
=======
        # self.object_list = []
>>>>>>> master
    def cal_center_point(self,box):

        return (int((box[2]-box[0])/2) + box[0],int((box[3]-box[1])/2) + box[1])

    def determine_within(self,frame_size, boxes):
        object_list = []
        if boxes is None:
            return object_list, self.region
        boxes=boxes.tolist()
        for k in range(len(boxes)):
            center = self.cal_center_point(boxes[k])
            for i in range(self.vertical):
                for j in range(self.horizontal):
                    if i * (1/self.vertical) * int(frame_size[1]) < center[0] < (i+1) * (1/self.vertical) * int(frame_size[1]) \
                            and j * (1/self.horizontal) * int(frame_size[0]) < center[1] < (j+1) * (1/self.horizontal) * int(frame_size[0]):
                        num = self.coor_region[(j,i)]
                        object_list.append(num)
        return object_list, self.region
<<<<<<< HEAD
=======
                        # print(len(boxes))
        # increase_frame = len(boxes) +1
        # self.region[num] +=increase_frame
        # all except region[num] -1


>>>>>>> master

    def drawlines(self,img,frame_shape):
        for i in range(self.horizontal - 1):
            cv2.line(img, (0, (i+1) * int(frame_shape[0] / self.horizontal)),
                     (frame_shape[1], (i+1) * int(frame_shape[0] / self.horizontal)),[0, 255, 255], 2)
        for j in range(self.vertical - 1):
            cv2.line(img, ((j + 1) * int(frame_shape[1] / self.vertical), 0),
                     ((j+1) * int(frame_shape[1] / self.vertical),frame_shape[0]), [0, 255, 255], 2)
