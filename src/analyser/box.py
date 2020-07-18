import torch
from src.utils.utils import cal_center_point

max_box_store = 50
box_ratio_thresh = 3
cal_hw_num = 10
hw_percent_ratio = 0.8


class Box:
    def __init__(self, box):
        self.boxes = box.unsqueeze(dim=0)
        self.centers = [cal_center_point(box)]

    def append(self, box):
        self.boxes = torch.cat([self.boxes, box.unsqueeze(dim=0)], dim=0)
        self.centers.append(cal_center_point(box))
        if len(self) > max_box_store:
            self.boxes = self.boxes[1:]
            self.centers = self.centers[1:]

    def __len__(self):
        return len(self.boxes)

    def cal_hw_ratio(self):
        tmp = self.boxes[-cal_hw_num:] if len(self) > cal_hw_num else self.boxes
        ratios = (tmp[:,3]-tmp[:,1])/(tmp[:,2]-tmp[:,0])
        print(ratios)
        return True if sum((ratios > box_ratio_thresh).float())/len(ratios) >= hw_percent_ratio else False

    def cal_curr_hw(self, box):
        h, w = box[3] - box[1], box[2] - box[0]
        # print("Height ")