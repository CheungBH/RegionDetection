import torch
from config import config
from yolo.preprocess import prep_frame
from yolo.util import dynamic_write_results
from yolo.darknet import Darknet


class ObjectDetectionYolo(object):
    def __init__(self, batchSize, cfg, weights):
        self.det_model = Darknet(cfg)
        self.det_model.load_weights(weights)
        self.det_model.net_info['height'] = config.input_size
        self.det_inp_dim = int(self.det_model.net_info['height'])
        assert self.det_inp_dim % 32 == 0
        assert self.det_inp_dim > 32
        self.det_model.cuda()
        self.det_model.eval()

        self.stopped = False
        self.batchSize = batchSize

    def __video_process(self, frame):
        img = []
        orig_img = []
        im_name = []
        im_dim_list = []
        img_k, orig_img_k, im_dim_list_k = prep_frame(frame, int(config.input_size))

        img.append(img_k)
        orig_img.append(orig_img_k)
        im_name.append('0.jpg')
        im_dim_list.append(im_dim_list_k)

        with torch.no_grad():
            # Human Detection
            img = torch.cat(img)
            im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
        return img, orig_img, im_name, im_dim_list

    def __get_bbox(self, img, orig_img, im_name, im_dim_list):
        with torch.no_grad():
            # Human Detection
            img = img.cuda()
            prediction = self.det_model(img, CUDA=True)
            # NMS process
            dets = dynamic_write_results(prediction, config.confidence,  config.num_classes, nms=True, nms_conf=config.nms_thresh)

            if isinstance(dets, int) or dets.shape[0] == 0:
                return orig_img[0], im_name[0], None, None, None, None, None

            dets = dets.cpu()
            im_dim_list = torch.index_select(im_dim_list, 0, dets[:, 0].long())
            scaling_factor = torch.min(self.det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

            # coordinate transfer
            dets[:, [1, 3]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
            dets[:, [2, 4]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

            dets[:, 1:5] /= scaling_factor
            for j in range(dets.shape[0]):
                dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
                dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
            boxes = dets[:, 1:5]
            scores = dets[:, 5:6]

        boxes_k = boxes[dets[:, 0] == 0]
        if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
            return orig_img[0], im_name[0], None, None, None, None, None
        inps = torch.zeros(boxes_k.size(0), 3, config.input_height, config.input_width)
        pt1 = torch.zeros(boxes_k.size(0), 2)
        pt2 = torch.zeros(boxes_k.size(0), 2)
        return orig_img[0], im_name[0], boxes_k, scores[dets[:, 0] == 0], inps, pt1, pt2

    def process(self, frame):
        img, orig_img, im_name, im_dim_list = self.__video_process(frame)
        inps, orig_img, boxes, _, scores, pt1, pt2 = self.__get_bbox(img, orig_img, im_name, im_dim_list)
        return inps, orig_img, boxes, scores, pt1, pt2
