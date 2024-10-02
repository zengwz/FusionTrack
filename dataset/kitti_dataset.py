import numpy as np
import re
from .kitti_data_base import *
import os
import pickle

class KittiDetectionDataset:
    def __init__(self,root_path):
        self.root_path = root_path
        self.velo_path = os.path.join(self.root_path,"velodyne")
        self.image_path = os.path.join(self.root_path,"image_2")
        self.calib_path = os.path.join(self.root_path,"calib")
        self.label_path = os.path.join(self.root_path,"label_2")

        self.all_ids = os.listdir(self.velo_path)

    def __len__(self):
        return len(self.all_ids)
    def __getitem__(self, item):

        name = str(item).zfill(6)

        velo_path = os.path.join(self.velo_path,name+'.bin')
        image_path = os.path.join(self.image_path, name+'.png')
        calib_path = os.path.join(self.calib_path, name+'.txt')
        label_path = os.path.join(self.label_path, name+".txt")

        P2,V2C = read_calib(calib_path)
        points = read_velodyne(velo_path,P2,V2C)
        image = read_image(image_path)
        labels,label_names = read_detection_label(label_path)
        labels[:,3:6] = cam_to_velo(labels[:,3:6],V2C)[:,:3]

        return P2,V2C,points,image,labels,label_names

class KittiTrackingDataset:
    def __init__(self,
                 data_path,
                 root_load_path,
                 det_2d_path,
                 seq_id,
                 type,
                 SF_thres_2d,
                 SF_thres_3d,
                 load_image=False,
                 load_points=False):
        self.seq_name = str(seq_id).zfill(4)
        self.data_path = data_path
        self.velo_path = os.path.join(self.data_path,"velodyne",self.seq_name)
        self.image_path = os.path.join(self.data_path,"image_02",self.seq_name)
        self.calib_path = os.path.join(self.data_path,"calib",self.seq_name)
        self.pose_path = os.path.join(self.data_path, "pose", self.seq_name,'pose.txt')
        self.type = type

        self.all_ids = os.listdir(self.velo_path)
        calib_path = self.calib_path + '.txt'

        self.P2, self.V2C = read_calib(calib_path)
        self.poses = read_pose(self.pose_path)
        self.load_image = load_image
        self.load_points = load_points

        self.det_2d_path = det_2d_path
        self.root_load_path = root_load_path
        self.SF_thres_2d = SF_thres_2d
        self.SF_thres_3d = SF_thres_3d


    def __len__(self):
        return len(self.all_ids)-1
    def __getitem__(self, item):

        frame_name = str(item).zfill(6)

        velo_path = os.path.join(self.velo_path, frame_name + '.bin')
        image_path = os.path.join(self.image_path, frame_name + '.png')

        if self.load_points:
            points = read_velodyne(velo_path,self.P2,self.V2C)
        else:
            points = None
        if self.load_image:
            image = read_image(image_path)
        else:
            image = None

        if item in self.poses.keys():
            pose = self.poses[item]
        else:
            pose = None

        det_2d_path = os.path.join(self.det_2d_path, self.seq_name, frame_name + '.txt')
        bboxes_3d_path = os.path.join(self.root_load_path, 'det_bboxes_3d', self.seq_name, frame_name + '.npy')
        features_path = os.path.join(self.root_load_path, 'features', self.seq_name, frame_name + '.npy')
        scores_path = os.path.join(self.root_load_path, 'det_scores', self.seq_name, frame_name + '.npy')
        if (os.path.exists(bboxes_3d_path)):
            det_3d = np.load(bboxes_3d_path)
            det_2d = self.parse_det_2d_from_txt(det_2d_path)
            det_scores = np.load(scores_path)
            features = np.load(features_path)
            if len(det_3d) != 0:
                SF_mask = det_scores > self.SF_thres_3d
                det_scores = det_scores[SF_mask]
                det_3d = det_3d[SF_mask]
                features = features[SF_mask]
        else:
            det_3d = np.zeros(shape=(0,7), dtype=float)
            det_2d = np.zeros(shape=(0,6), dtype=float)
            det_scores = np.zeros(shape=(0),dtype=float)
            features = None

        return self.P2, self.V2C, pose, det_3d, det_2d, features, det_scores, [self.seq_name, frame_name], points, image
    
    def parse_det_2d_from_pickle(self, pickle_path):
        list_dets2d = []
        print(f"Parsing {pickle_path}")
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        data = data['det_bboxes']
        for i, img_dets2d in enumerate(data):
            dets2d = np.zeros((0, 6), dtype=np.float32)
            img_dets2d = img_dets2d[0]
            for cate_id, dets in enumerate(img_dets2d):
                # cate_id_col = np.full((dets.shape[0], 1), cate_id, dtype=np.float32)
                dets = np.append(dets, [0.0]).reshape((1, -1))
                dets2d = np.concatenate((dets2d, dets), axis=0)
            dets2d = dets2d[dets2d[:, 4] > 0.0]
            list_dets2d.append(dets2d)
        return list_dets2d
    
    def parse_det_2d_from_txt(self, det_2d_path):
        det_2d_list = np.zeros((0,6), dtype=np.float32)
        with open(det_2d_path) as f:
            det_2d_lines = f.readlines()
            for id, det_2d_str in enumerate(det_2d_lines):
                det_2d = det_2d_str.rstrip().split(' ')
                det_2d = np.array(det_2d, dtype=np.float32).reshape((-1, 6))
                if det_2d[0, 4] > self.SF_thres_2d:
                    det_2d_list = np.concatenate((det_2d_list, det_2d), axis=0)
        return np.array(det_2d_list)

def nms(detections, iou_threshold):
    # 按照得分降序排序
    scores = detections[:, 4]
    sorted_indices = np.argsort(scores)[::-1]
    sorted_detections = detections[sorted_indices]

    keep = []
    while len(sorted_detections) > 0:
        # 选择具有最高分数的检测框
        best_detection = sorted_detections[0]
        keep.append(best_detection)

        # 计算与其余检测框的IoU
        box = best_detection[:4]
        other_boxes = sorted_detections[1:, :4]
        iou = calculate_iou(box, other_boxes)

        # 保留IoU低于阈值的检测框
        indices_to_keep = np.where(iou <= iou_threshold)[0]
        sorted_detections = sorted_detections[indices_to_keep + 1]

    return np.stack(keep)

def calculate_iou(box, other_boxes):
    # 计算两个框的交集区域
    x1 = np.maximum(box[0], other_boxes[:, 0])
    y1 = np.maximum(box[1], other_boxes[:, 1])
    x2 = np.minimum(box[2], other_boxes[:, 2])
    y2 = np.minimum(box[3], other_boxes[:, 3])

    intersection_area = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)

    # 计算两个框的并集区域
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    other_boxes_area = (other_boxes[:, 2] - other_boxes[:, 0] + 1) * (other_boxes[:, 3] - other_boxes[:, 1] + 1)
    union_area = box_area + other_boxes_area - intersection_area

    # 计算IoU
    iou = intersection_area / union_area

    return iou  