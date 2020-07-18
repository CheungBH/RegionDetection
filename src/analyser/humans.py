from .people import Person
import cv2
from config.config import frame_size
import numpy as np


class HumanProcessor:
    def __init__(self):
        self.stored_id = []
        self.PEOPLE = {}
        self.curr_id = []
        self.untracked_id = []

    def clear(self):
        self.curr_id = []
        self.untracked_id = []

    def update_box(self, id2box):
        self.clear()
        for k, v in id2box.items():
            self.curr_id.append(k)
            if k not in self.stored_id:
                self.PEOPLE[k] = Person(k, v)
                self.stored_id.append(k)
            else:
                self.PEOPLE[k].BOX.append(v)
            self.PEOPLE[k].BOX.cal_size_ratio()

    def update_untracked(self):
        self.untracked_id = [x for x in self.stored_id if x not in self.curr_id]
        for x in self.untracked_id:
            self.PEOPLE[x].disappear += 1

    def update(self, id2box):
        self.update_box(id2box)
        self.update_untracked()
        # self.vis_box_size()

    def vis_box_size(self, im_box):
        img_cnt = cv2.imread("src/black.jpg")
        img_cnt = cv2.resize(img_cnt, frame_size)
        curr = sorted(self.curr_id)
        for num, idx in enumerate(curr):
            self.PEOPLE[idx].BOX.vis_box_size(img_cnt, idx, num)
            h, w = self.PEOPLE[idx].BOX.cal_curr_hw()
            cv2.putText(im_box, "{}".format(round((h/w).tolist(), 4)), self.PEOPLE[idx].BOX.curr_center(),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        # cv2.imshow("box size", img_black)
        im_box = cv2.resize(im_box, frame_size)
        return np.concatenate((img_cnt, im_box), axis=0)