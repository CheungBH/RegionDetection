from .box import Box
from .keypoint import Keypoint

max_disappear = 10
max_model_pred = 5


class Person:
    def __init__(self, idx, box):
        self.id = idx
        self.BOX = Box(box)
        self.KPS = Keypoint()
        self.disappear = 0
        self.img = []
        self.RNN_pred = []

    def box_len(self):
        return len(self.BOX)

    def kps_len(self):
        return len(self.KPS)

    def update_disappear(self, flag):
        if flag == 1:
            self.disappear = 0
        elif flag == 0:
            if self.disappear < max_disappear:
                self.disappear += 1

    def update_RNN_pred(self, pred):
        self.RNN_pred.append(pred)
        if len(self.RNN_pred) >= max_model_pred:
            self.RNN_pred = self.RNN_pred[-max_model_pred:]

