import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from collections import defaultdict

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from tracker.byte_tracker import MCBYTETracker
from utils.visualize import plot_tracking_mc


def track(save_img=True):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    half = False
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    args = {
        'track_thresh': 0.5,
        'match_thresh': 0.8,
        'n_classes': 2,
        'class_names': [],
        'track_buffer': 30,
        'with_reid': True,
        'fast_reid_config': 'weights/sbs_R50_person_big/config.yaml',
        'fast_reid_weights': 'weights/sbs_R50_person_big/model_best.pth',
        'device': 'cuda'
    }
    tracker = MCBYTETracker(args, frame_rate=25)

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 1, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    id2cls = defaultdict(str)
    cls2id = defaultdict(int)
    for cls_id, cls_name in names.items():
        id2cls[cls_id] = cls_name
        cls2id[cls_name] = cls_id

    seq_filename = Path(source).name + '.txt'
    file_result = open(str(save_dir / seq_filename), 'w')

    frame_id = 0
    for path, img, im0, vid_cap in dataset:
        frame_id += 1
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        pred[0][:, :4] = scale_coords(img.shape[2:], pred[0][:, :4], im0.shape[:2])
        _pred = pred[0].cpu().numpy()
        # if len(pred[0]):
        online_dict = tracker.update(_pred, im0)
        online_tlwhs_dict = defaultdict(list)
        online_tr_ids_dict = defaultdict(list)
        for cls_id in range(tracker.n_classes):  # process each object class
            online_targets = online_dict[cls_id]
            for track in online_targets:
                online_tlwhs_dict[cls_id].append(track.tlwh)
                # online_tlwhs_dict[cls_id].append(track.get_tlwh())
                online_tr_ids_dict[cls_id].append(track.track_id)
                result_frame = f'{frame_id}, {track.track_id}, {track.tlwh[0]}, {track.tlwh[1]}, {track.tlwh[2]}, {track.tlwh[3]}, {cls_id}'
                file_result.write(result_frame+'\n')

        show_img = np.repeat(im0, 3, axis=2)
        online_img = plot_tracking_mc(img=show_img,
                                      tlwhs_dict=online_tlwhs_dict,
                                      obj_ids_dict=online_tr_ids_dict,
                                      num_classes=tracker.n_classes,
                                      frame_id=frame_id,
                                      fps=0.0,
                                      id2cls=id2cls)


        # else:
        #     online_img = im0s
        # det = pred[0]
        # if len(det):
        #     for *xyxy, conf, cls in reversed(det):
        #         if save_img or view_img:  # Add bbox to image
        #             label = f'{names[int(cls)]} {conf:.2f}'
        #             plot_one_box(xyxy, online_img, label=label, color=colors[int(cls)], line_thickness=1)

        if save_img:
            p = Path(path)  # to Path
            save_path = str(save_dir / p.name)
            # print(save_path)
            cv2.imwrite(save_path, online_img)


def track_all(save_img=True):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    half = False
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    sequences = sorted(os.listdir(source))
    for seq in sequences:
        print(seq)
        args = {
            'track_thresh': 0.5,
            'match_thresh': 0.8,
            'n_classes': 2,
            'class_names': [],
            'track_buffer': 30,
            'with_reid': True,
            'fast_reid_config': '/home/tungpt37/Workspace/tungpt37/Model/Detection/yolov7-MOT-ReID/fast_reid/logs/camel/sbs_R50_person_big/config.yaml',
            'fast_reid_weights': '/home/tungpt37/Workspace/tungpt37/Model/Detection/yolov7-MOT-ReID/fast_reid/logs/camel/sbs_R50_person_big/model_best.pth',
            'device': 'cuda'
        }
        tracker = MCBYTETracker(args, frame_rate=25)

        # Dataloader
        dataset = LoadImages(os.path.join(source, seq), img_size=imgsz, stride=stride)

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 1, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        id2cls = defaultdict(str)
        cls2id = defaultdict(int)
        for cls_id, cls_name in names.items():
            id2cls[cls_id] = cls_name
            cls2id[cls_name] = cls_id

        seq_filename = Path(seq).name + '.txt'
        file_result = open(str(save_dir / seq_filename), 'w')

        frame_id = 0
        for path, img, im0, vid_cap in dataset:
            frame_id += 1
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

            pred[0][:, :4] = scale_coords(img.shape[2:], pred[0][:, :4], im0.shape[:2])
            _pred = pred[0].cpu().numpy()
            # if len(pred[0]):
            online_dict = tracker.update(_pred, im0)
            online_tlwhs_dict = defaultdict(list)
            online_tr_ids_dict = defaultdict(list)
            # for cls_id in range(tracker.n_classes):  # process each object class
            for cls_id in [0, 1]:  # process each object class
                online_targets = online_dict[cls_id]
                for track in online_targets:
                    online_tlwhs_dict[cls_id].append(track.tlwh)
                    # online_tlwhs_dict[cls_id].append(track.get_tlwh())
                    online_tr_ids_dict[cls_id].append(track.track_id)
                    # if cls_id == 1:
                    #     cls_id += 1
                    result_frame = f'{frame_id}, {track.track_id}, {track.tlwh[0]}, {track.tlwh[1]}, {track.tlwh[2]}, {track.tlwh[3]}, {cls_id + 1}'
                    file_result.write(result_frame+'\n')

            # online_img = plot_tracking_mc(img=im0s,
            #                               tlwhs_dict=online_tlwhs_dict,
            #                               obj_ids_dict=online_tr_ids_dict,
            #                               num_classes=tracker.n_classes,
            #                               frame_id=frame_id,
            #                               fps=0.0,
            #                               id2cls=id2cls)

            # else:
            #     online_img = im0s
            # det = pred[0]
            # if len(det):
            #     for *xyxy, conf, cls in reversed(det):
            #         if save_img or view_img:  # Add bbox to image
            #             label = f'{names[int(cls)]} {conf:.2f}'
            #             plot_one_box(xyxy, online_img, label=label, color=colors[int(cls)], line_thickness=1)

            # if save_img:
            #     p = Path(path)  # to Path
            #     save_path = str(save_dir / p.name)
            #     print(save_path)
            #     cv2.imwrite(save_path, online_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        # if opt.update:  # update all models (to fix SourceChangeWarning)
        #     for opt.weights in ['yolov7.pt']:
        #         detect()
        #         strip_optimizer(opt.weights)
        # else:
        # track_all()
        track()
