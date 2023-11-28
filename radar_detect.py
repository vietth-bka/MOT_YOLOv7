import argparse
import os

import cv2
import torch
import numpy as np
from numpy import random
from collections import defaultdict

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box

from tracker.byte_tracker import MCBYTETracker
from utils.visualize import plot_tracking_mc


class YOLODetector:
    def __init__(self, weights, imgsz, conf_thres=0.25, iou_thres=0.4, device=torch.device("cuda")):
        self.org_size = None
        self.device = device
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.model = attempt_load(weights, map_location=device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride

        self.model(torch.zeros(1, 1, imgsz, imgsz).to(device).type_as(next(self.model.parameters())))
        self.img_size = (imgsz, imgsz)
        self.names = self.model.names

    def preprocess(self, input_image, img_size, stride):
        self.org_size = input_image.shape
        im = letterbox(input_image, img_size, stride=stride)[0]
        self.img_size = im.shape
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = np.expand_dims(im, axis=0)
        im = torch.from_numpy(im).to(self.device)
        im = im.float()
        im /= 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)

        return im

    def post_process(self, prediction):
        _pred = non_max_suppression(prediction, self.conf_thres, self.iou_thres)
        _pred = _pred[0]
        _pred[:, :4] = scale_coords(self.img_size, _pred[:, :4], self.org_size[:2])
        _pred = _pred.detach().cpu().numpy()
        return _pred

    def run(self, img):
        im = self.preprocess(img, self.img_size, self.stride)
        y = self.model(im)
        out = self.post_process(y[0])
        return out


def detect(weights, source):
    detector = YOLODetector(weights, 416)
    names = detector.names
    print(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    args = {
        'track_thresh': 0.5,
        'match_thresh': 0.8,
        'n_classes': 6,
        'class_names': [],
        'track_buffer': 30
    }
    tracker = MCBYTETracker(args, frame_rate=1)

    id2cls = defaultdict(str)
    cls2id = defaultdict(int)
    for cls_id, cls_name in names.items():
        id2cls[cls_id] = cls_name
        cls2id[cls_name] = cls_id

    subdirs = sorted(os.listdir(source))[13:20]
    n_view = len(subdirs)

    # filepath_dict = defaultdict(list)   # type: dict[int, list[str]]
    filepath_arr = []

    for i, subdir in enumerate(subdirs):
        view_path = os.path.join(source, subdir)
        filenames = sorted(os.listdir(view_path))
        arr_ = []
        for filename in filenames:
            # filepath_dict[i].append(os.path.join(view_path, filename))
            arr_.append(os.path.join(view_path, filename))
        filepath_arr.append(arr_)

    filepath_arr = np.asarray(filepath_arr)

    n_frame = filepath_arr.shape[1]
    # n_view = 4
    h, w = None, None
    frame_id = 0
    for t_index in range(n_frame):

        t_pred = np.empty((0, 6))
        all_image = None
        for v_index in range(n_view):
            img = cv2.imread(filepath_arr[v_index][t_index])
            if h is None and w is None:
                h, w = img.shape[:2]
            pred = detector.run(img)
            pred[:, 0] += v_index*w
            pred[:, 2] += v_index*w

            t_pred = np.append(t_pred, pred, axis=0)

            if all_image is None:
                all_image = img.copy()
            else:
                all_image = np.concatenate([all_image, img], axis=1)

        # if len(t_pred):
        #     for *xyxy, conf, cls in reversed(t_pred):
        #         label = f'{names[int(cls)]} {conf:.2f}'
        #         plot_one_box(xyxy, all_image, label=label, color=colors[int(cls)], line_thickness=1)

        online_dict = tracker.update(t_pred)
        online_tlwhs_dict = defaultdict(list)
        online_tr_ids_dict = defaultdict(list)
        for cls_id in range(tracker.n_classes):  # process each object class
            online_targets = online_dict[cls_id]
            for track in online_targets:
                online_tlwhs_dict[cls_id].append(track.tlwh)
                # online_tlwhs_dict[cls_id].append(track.get_tlwh())
                online_tr_ids_dict[cls_id].append(track.track_id)

        online_img = plot_tracking_mc(img=all_image,
                                      tlwhs_dict=online_tlwhs_dict,
                                      obj_ids_dict=online_tr_ids_dict,
                                      num_classes=tracker.n_classes,
                                      frame_id=frame_id + 1,
                                      fps=0.0,
                                      id2cls=id2cls)
        frame_id += 1
            # if v_index == 20:
        cv2.imwrite(f"runs/track/t5/test_{t_index}.jpg", online_img)


detect("runs/train/tod6c24/weights/last.pt", "/home/tungpt37/Workspace/tungpt37/Data/Thermal_Radar/Data01")
