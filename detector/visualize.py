import cv2
from config import config

water_top = config.water_top


class BBoxVisualizer(object):
    def __init__(self):
        self.black_color = (0, 0, 255)
        self.gray_color = (255, 0, 0)
        self.origin_color = (0, 255, 0)

    def visualize_black(self, bboxes, img):
        for bbox in bboxes:
            # img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.black_color, 2)
            img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), self.black_color, 2)
        return img

    def visualize_dip(self, bboxes, img):
        for bbox in bboxes:
            img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), self.origin_color, 2)
        return img

    def visualize_gray(self, bboxes, img):
        for bbox in bboxes:
            img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), self.gray_color, 2)
        return img
