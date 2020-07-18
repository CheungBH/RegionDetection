import torch
from src.utils.utils import cal_center_point
import cv2

max_box_store = 50
box_ratio_thresh = 3
cal_hw_num = 10
hw_percent_ratio = 0.8


class Box:
    def __init__(self, box):
        self.boxes = box.unsqueeze(dim=0)
        self.centers = [cal_center_point(box)]
        self.ratios = torch.FloatTensor([])
        self.curr_box = box

    def append(self, box):
        self.curr_box = box
        self.boxes = torch.cat([self.boxes, box.unsqueeze(dim=0)], dim=0)
        self.centers.append(cal_center_point(box))
        if len(self) > max_box_store:
            self.boxes = self.boxes[1:]
            self.centers = self.centers[1:]

    def __len__(self):
        return len(self.boxes)

    def cal_size_ratio(self):
        tmp = self.boxes[-cal_hw_num:] if len(self) > cal_hw_num else self.boxes
        self.ratios = (tmp[:,3]-tmp[:,1])/(tmp[:,2]-tmp[:,0])
        # print(self.ratios)
        return True if sum((self.ratios > box_ratio_thresh).float())/len(self.ratios) >= hw_percent_ratio else False

    def cal_curr_hw(self):
        h, w = self.curr_box[3] - self.curr_box[1], self.curr_box[2] - self.curr_box[0]
        return h, w

    def text_color(self, r):
        return (0, 255, 255) if r > box_ratio_thresh else (255, 255, 0)

    def curr_center(self):
        return cal_center_point(self.curr_box)

    def vis_box_size(self, img, idx, num):
        cv2.putText(img, "id{}".format(idx), (30 + 250*num, 40), cv2.FONT_ITALIC, 1, (0, 255, 255), 3)
        for i, item in enumerate(self.ratios.tolist()[::-1]):
            cv2.putText(img, "f-{}: {}".format(i, round(item, 2)), (30 + 250*num, + 100+ 40*i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.text_color(item), 1)