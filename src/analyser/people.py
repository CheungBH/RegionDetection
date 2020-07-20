from .box import Box
from .keypoint import Keypoint
max_disappear = 10


class Person:
    def __init__(self, idx, box):
        self.id = idx
        self.BOX = Box(box)
        self.KPS = Keypoint()
        self.pred = []
        self.disappear = 0
        self.img = []

    def box_len(self):
        return len(self.BOX)

    def kps_len(self):
        return len(self.KPS.kps)

    def update_disappear(self, flag):
        if flag == 1:
            self.disappear = 0
        elif flag == 0:
            if self.disappear < max_disappear:
                self.disappear += 1

