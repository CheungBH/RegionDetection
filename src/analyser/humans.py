from .people import Person
import cv2
from config.config import frame_size


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
            print(self.PEOPLE[k].BOX.cal_size_ratio())

    def update_untracked(self):
        self.untracked_id = [x for x in self.stored_id if x not in self.curr_id]
        for x in self.untracked_id:
            self.PEOPLE[x].disappear += 1

    def update(self, id2box):
        self.update_box(id2box)
        self.update_untracked()
        # self.vis_box_size()

    def vis_box_size(self, fr):
        img_black = cv2.imread("src/black.jpg")
        for num, idx in enumerate(self.curr_id):
            self.PEOPLE[idx].BOX.vis_box_size(img_black, idx, num)

        # cv2.imshow("box size", img_black)
        return cv2.resize(img_black, frame_size)
